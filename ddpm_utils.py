import numpy as np
import torch
import math


# def normal_kl(mean1, logvar1, mean2, logvar2):
#   """
#   KL divergence between normal distributions parameterized by mean and log-variance.
#   """
#   return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
#                 + tf.squared_difference(mean1, mean2) * tf.exp(-logvar2))
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的 KL 散度
    """
    assert mean1.shape == logvar1.shape == mean2.shape == logvar2.shape
    
    # 计算 KL 散度
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                + torch.pow(mean1 - mean2, 2) * torch.exp(-logvar2))
    
    return kl


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


def meanflat(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).mean(dim=1)


def approx_standard_normal_cdf(x):
  return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))



def discretized_gaussian_log_likelihood(x: torch.Tensor, *, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """
    Discretized Gaussian likelihood
    假设输入 x ∈ [-1, 1]，原始数据是 [0,255] 归一化而来
    """
    assert x.shape == means.shape == log_scales.shape, f"Shape mismatch: {x.shape}, {means.shape}, {log_scales.shape}"

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)

    # 对应 (x + 1/255)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    # 对应 (x - 1/255)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)

    # 三种情况
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1. - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    # 拼接最终结果
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(torch.clamp(cdf_delta, min=1e-12))
        )
    )

    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion2:
  """
  Contains utilities for the diffusion model.

  Arguments:
  - what the network predicts (x_{t-1}, x_0, or epsilon)
  - which loss function (kl or unweighted MSE)
  - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
  - what type of decoder, and how to weight its loss? is its variance learned too?
  """

  def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
    self.model_mean_type = model_mean_type  # xprev, xstart, eps
    self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
    self.loss_type = loss_type  # kl, mse

    assert isinstance(betas, np.ndarray)
    self.betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
    assert (betas > 0).all() and (betas <= 1).all()
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    alphas = 1. - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
    assert self.alphas_cumprod_prev.shape == (timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
    self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    a_tensor = torch.as_tensor(a, dtype=torch.float32, device=t.device)
    out = a_tensor[t]  # 使用索引提取
    assert out.shape == (bs,)
    return out.view(*([bs] + ((len(x_shape) - 1) * [1])))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = torch.randn_like(x_start)
    assert noise.shape == x_start.shape
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    posterior_mean = (
        self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
            x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  # def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
  #   B, H, W, C = x.shape
  #   assert t.shape == [B]
  #   model_output = denoise_fn(x, t)

  #   # Learned or fixed variance?
  #   if self.model_var_type == 'learned':
  #     assert model_output.shape == [B, H, W, C * 2]
  #     model_output, model_log_variance = torch.split(model_output, model_output.size(-1) // 2, dim=-1)
  #     model_variance = torch.exp(model_log_variance)
  #   elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
  #     # below: only log_variance is used in the KL computations
  #     model_variance, model_log_variance = {
  #       # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
  #       'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
  #       'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
  #     }[self.model_var_type]
  #     model_variance = self._extract(model_variance, t, x.shape) * torch.ones_like(x)
  #     model_log_variance = self._extract(model_log_variance, t, x.shape) * torch.ones_like(x)
  #   else:
  #     raise NotImplementedError(self.model_var_type)

  #   # Mean parameterization
  #   _maybe_clip = lambda x_: (torch.clamp(x_, -1., 1.) if clip_denoised else x_)
  #   if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
  #     pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
  #     model_mean = model_output
  #   elif self.model_mean_type == 'xstart':  # the model predicts x_0
  #     pred_xstart = _maybe_clip(model_output)
  #     model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
  #   elif self.model_mean_type == 'eps':  # the model predicts epsilon
  #     pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
  #     model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
  #   else:
  #     raise NotImplementedError(self.model_mean_type)

  #   assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
  #   if return_pred_xstart:
  #     return model_mean, model_variance, model_log_variance, pred_xstart
  #   else:
  #     return model_mean, model_variance, model_log_variance


  def p_mean_variance(self, model_output, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
    B, C, H, W = x.shape
    assert t.shape == (B,)
    # model_output = denoise_fn(x, t)

    # Learned or fixed variance?
    if self.model_var_type == 'learned':
      assert model_output.shape == (B, H, W, C * 2)
      model_output, model_log_variance = torch.split(model_output, model_output.size(-1) // 2, dim=-1)
      model_variance = torch.exp(model_log_variance)
    elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
      # below: only log_variance is used in the KL computations
      model_variance, model_log_variance = {
        # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
        'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
        'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
      }[self.model_var_type]
      model_variance = self._extract(model_variance, t, x.shape) * torch.ones_like(x)
      model_log_variance = self._extract(model_log_variance, t, x.shape) * torch.ones_like(x)
    else:
      raise NotImplementedError(self.model_var_type)

    # Mean parameterization
    _maybe_clip = lambda x_: (torch.clamp(x_, -1., 1.) if clip_denoised else x_)
    if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
      pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
      model_mean = model_output
    elif self.model_mean_type == 'xstart':  # the model predicts x_0
      pred_xstart = _maybe_clip(model_output)
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    elif self.model_mean_type == 'eps':  # the model predicts epsilon
      pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    else:
      raise NotImplementedError(self.model_mean_type)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    if return_pred_xstart:
      return model_mean, model_variance, model_log_variance, pred_xstart
    else:
      return model_mean, model_variance, model_log_variance


  def _predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        self._extract(torch.tensor(self.sqrt_recip_alphas_cumprod, device=x_t.device), t, x_t.shape) * x_t -
        self._extract(torch.tensor(self.sqrt_recipm1_alphas_cumprod, device=x_t.device), t, x_t.shape) * eps
    )

  def _predict_xstart_from_xprev(self, x_t, t, xprev):
    assert x_t.shape == xprev.shape
    return (  # (xprev - coef2*x_t) / coef1
        self._extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
        self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
    )

  # === Sampling ===

  def p_sample(self, model_output, *, x, t, noise_fn, clip_denoised=True, return_pred_xstart: bool):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      model_output, x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    noise = noise_fn(x.shape, dtype=x.dtype, device=x.device)
    assert noise.shape == x.shape
    # no noise when t == 0
    nonzero_mask = (t != 0).to(device=x.device, dtype=x.dtype).view(*([x.shape[0]] + [1] * (len(x.shape) - 1)))
    sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    assert sample.shape == pred_xstart.shape
    return (sample, pred_xstart) if return_pred_xstart else sample

  def p_sample_loop(self, denoise_fn, *, shape, noise_fn=torch.randn):
    """
    Generate samples
    """
    assert isinstance(shape, (tuple, list))
    img_0 = noise_fn(shape=shape, dtype=torch.float32)
    img = img_0
    for i in reversed(range(self.num_timesteps)):
        t = torch.full((shape[0],), i, dtype=torch.long, device=img.device)
        img = self.p_sample(
            denoise_fn=denoise_fn,
            x=img,
            t=t,
            noise_fn=noise_fn,
            return_pred_xstart=False
        )
    img_final = img
    assert img_final.shape == shape
    return img_final

  def p_sample_loop_progressive(self, denoise_fn, *, shape, noise_fn=torch.randn, include_xstartpred_freq=50):
    """
    Generate samples and keep track of prediction of x0
    """
    # 初始时间步和初始噪声
    i_0 = self.num_timesteps - 1
    img = noise_fn(shape)
    
    # 计算要记录的次数
    num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
    xstartpreds = torch.zeros((shape[0], num_recorded_xstartpred, *shape[1:]), dtype=torch.float32)
    
    # 逆序循环
    with torch.no_grad():
        for i in reversed(range(self.num_timesteps)):
            # 当前时间步 t
            t = torch.full((shape[0],), i, dtype=torch.long)
            
            # 调用 p_sample 获取 sample 和预测 x0
            sample, pred_xstart = self.p_sample(
                denoise_fn=denoise_fn,
                x=img,
                t=t,
                noise_fn=noise_fn,
                return_pred_xstart=True
            )
            
            # 更新 img
            img = sample
            
            # 如果当前时间步是要记录的 checkpoint，更新 xstartpreds
            record_idx = i // include_xstartpred_freq
            if record_idx < num_recorded_xstartpred:
                xstartpreds[:, record_idx] = pred_xstart
    
    return img, xstartpreds

  # === Log likelihood calculation ===

  def _vb_terms_bpd(self, model_output, x_start, x_t, t, *, clip_denoised: bool, return_pred_xstart: bool):
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      model_output, x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
    kl = meanflat(kl) / np.log(2.)

    decoder_nll = -discretized_gaussian_log_likelihood(
      x_start, means=model_mean, log_scales=0.5 * model_log_variance)
    assert decoder_nll.shape == x_start.shape
    decoder_nll = meanflat(decoder_nll) / np.log(2.)

    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    assert kl.shape == decoder_nll.shape == t.shape == [x_start.shape[0]]
    output = torch.where(t == 0, decoder_nll, kl)
    return (output, pred_xstart) if return_pred_xstart else output

  def training_losses(self, model_output, x_start, t, noise=None):
    """
    Training loss calculation
    """

    # Add noise to data
    assert t.shape == [x_start.shape[0]]
    if noise is None:
      noise = torch.randn(x_start.shape, dtype=x_start.dtype, device=x_start.device)
    assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Calculate the loss
    if self.loss_type == 'kl':  # the variational bound
      losses = self._vb_terms_bpd(
        model_output=model_output, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
    elif self.loss_type == 'mse':  # unweighted MSE
      assert self.model_var_type != 'learned'
      target = {
        'xprev': self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
        'xstart': x_start,
        'eps': noise
      }[self.model_mean_type]
      assert model_output.shape == target.shape == x_start.shape
      losses = meanflat(torch.pow(target - model_output, 2))
    else:
      raise NotImplementedError(self.loss_type)

    assert losses.shape == t.shape
    return losses

  def _prior_bpd(self, x_start):
    B, T = x_start.shape[0], self.num_timesteps
    qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=torch.full((B,), T - 1, dtype=torch.int32, device=x_start.device))
    kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
    assert kl_prior.shape == x_start.shape
    return meanflat(kl_prior) / np.log(2.)

  def calc_bpd_loop(self, denoise_fn, x_start, *, clip_denoised=True):
    """
    PyTorch 等价实现：
    - x_start: tensor with shape (B, H, W, C)  （与 TF 版本保持同布局）
    - returns: total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt
    """
    B = x_start.shape[0]
    T = self.num_timesteps

    # 初始化存储每一步的 term / mse，shape [B, T]
    terms_bt = torch.zeros((B, T), dtype=x_start.dtype, device=x_start.device)
    mse_bt = torch.zeros((B, T), dtype=x_start.dtype, device=x_start.device)

    # 在推理时不计算梯度（TensorFlow 原版用了 back_prop=False）
    with torch.no_grad():
        # 逆序 t = T-1 ... 0
        for t in reversed(range(T)):
            # t_b shape [B], dtype long
            t_b = torch.full((B,), t, dtype=torch.long, device=x_start.device)

            # 计算 x_t = q_sample(x_start, t_b)
            x_t = self.q_sample(x_start=x_start, t=t_b)

            # 计算当前 timestep 的 VLB term，并获得 pred_xstart
            # _vb_terms_bpd(..., return_pred_xstart=True) 返回 (vals_b, pred_xstart)
            new_vals_b, pred_xstart = self._vb_terms_bpd(
                denoise_fn,
                x_start=x_start,
                x_t=x_t,
                t=t_b,
                clip_denoised=clip_denoised,
                return_pred_xstart=True
            )
            # new_vals_b: shape [B], pred_xstart: shape same as x_start (B,H,W,C)

            # 逐样本 MSE between pred_xstart and x_start (mean over non-batch dims)
            new_mse_b = meanflat((pred_xstart - x_start) ** 2)  # shape [B]

            # 将当前计算的值写入对应的时间步列
            terms_bt[:, t] = new_vals_b
            mse_bt[:, t] = new_mse_b

    # prior term
    prior_bpd_b = self._prior_bpd(x_start)  # shape [B]

    total_bpd_b = terms_bt.sum(dim=1) + prior_bpd_b  # shape [B]

    assert terms_bt.shape == mse_bt.shape == (B, T)
    assert total_bpd_b.shape == prior_bpd_b.shape == (B,)

    return total_bpd_b, terms_bt, prior_bpd_b, mse_bt

def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        device = x_target_noisy.device
        B,N,_,H,W = x_target_noisy.shape
        # denoise
        with torch.no_grad():
            a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1,1)
            a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1,1)
            sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1,1)
            sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1,1)

            pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
            dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            # add the option to set a start_step
            if index>self.start_step:
                noise = sigma_t * torch.randn_like(x_target_noisy)
                x_prev = x_prev + noise
            # Directing the clip embedding of the image to the clip embedding of the reference.
            else:
                lr_schedule = torch.linspace(self.lr_start, self.lr_end, self.ddpm_num_timesteps)
                curr_lr = lr_schedule[index]
                if not is_step0:
                    for b in range(B):
                        anchor = random.randint(0, N-1)
                        with torch.no_grad():
                            x_prev_decoded = torch.stack([self.model.decode_first_stage(x_prev[:, ni]) for ni in range(N)], 1)
                            x_prev_decoded = torch.clamp(x_prev_decoded, max=1.0, min=-1.0)
                            reference_embed = self.clip_model.forward(x_prev_decoded[:, anchor])

                        for n in range(N):
                            if n!=anchor:
                                x_n = x_prev[:,n].clone().detach().requires_grad_()
                                optimizer = torch.optim.Adam([x_n], lr=curr_lr)
                                for i in range(3):
                                    optimizer.zero_grad()
                                    x_n_decoded = self.model.decode_first_stage(x_n)
                                    if(not x_n_decoded.requires_grad): print("detached! after self.model.decode_first_stage")
                                    x_n_decoded = torch.clamp(x_n_decoded, max=1.0, min=-1.0)

                                    prevn_embed = self.clip_model.forward(x_n_decoded)
                                    if(not prevn_embed.requires_grad): print("detached! after self.clip_model.forward(x_n_decoded)")
                                    
                                    loss = -torch.cosine_similarity(reference_embed, prevn_embed).mean()
                                    loss.backward()
                                    optimizer.step()
                                x_prev[:,n] = x_n
        return x_prev
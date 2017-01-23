function  [im_out,par]    =   Denoising(par)
% im_in = par.nim;
im_out    =   par.nim;
par.nSig0 = par.nSig;
% parameters for noisy image
[h,  w]      =  size(im_out);
par.maxr = h-par.ps+1;
par.maxc = w-par.ps+1;
par.h = h;
par.w = w;
r          =  1:par.step:par.maxr;
par.r          =  [r r(end)+1:par.maxr];
c          =  1:par.step:par.maxc;
par.c          =  [c c(end)+1:par.maxc];
par.lenr = length(par.r);
par.lenc = length(par.c);
par.ps2 = par.ps^2;
par.maxrc = par.maxr*par.maxc;
par.lenrc = par.lenr*par.lenc;
for ite  =  1 : par.IteNum  
    % iterative regularization
    im_out = im_out+par.delta*(par.nim-im_out);
    % searching  non-local patches
    [nDCnlX, blk_arr, DC, Sigma, par] = CalNonLocal( im_out, par);
    % PG-GMM training
    if mod(ite-1,par.changeD)==0
        [model,~,cls_idx] = emgm(nDCnlX,par);
        % cluster segmentation
        [idx,  s_idx]    =  sort(cls_idx);
        idx2   =  idx(1:end-1) - idx(2:end);
        seq    =  find(idx2);
        seg    =  [0; seq; length(cls_idx)];
        par.maxiter = par.maxiter + 5;
    end
    % estimation of noise variance
    if ite==1
        Sigma = par.nSig0 * ones(size(Sigma));
    end
    % Weighted Sparse Coding
    X_hat = zeros(par.ps^2, par.maxr*par.maxc, 'single');
    W = zeros(par.ps^2, par.maxr*par.maxc, 'single');
    for i = 1:length(seg)-1
        idx    =   s_idx(seg(i)+1:seg(i+1));
        cls    =   cls_idx(idx(1));
        [GMM_D, GMM_S, ~] = svd(model.covs(:,:,cls));
        GMM_S = diag(GMM_S);
        for j=1:size(idx, 1)
            Y         =   nDCnlX(:, (idx(j)-1)*par.nlsp+1:idx(j)*par.nlsp);
            lambdaM = bsxfun(@rdivide, par.lambda*Sigma(blk_arr(:,idx(j))) .^2, sqrt(GMM_S) + eps );
            % soft threshold
            B = GMM_D' * Y;
            Alpha = sign(B) .* max(abs(B) - lambdaM, 0);
            % Recovered Estimated Patches
            Yhat = GMM_D * Alpha;
            % add DC components and aggregation
            X_hat(:, blk_arr(:,idx(j))) = X_hat(:, blk_arr(:,idx(j)))+bsxfun(@plus, Yhat, DC(:,idx(j)));
            W(:, blk_arr(:,idx(j))) = W(:, blk_arr(:,idx(j)))+ones(par.ps^2, par.nlsp);
        end
    end
    % Reconstruction
    im_out   =  zeros(h,w,'single');
    im_wei   =  zeros(h,w,'single');
    r = 1:par.maxr;
    c = 1:par.maxc;
    k = 0;
    for i = 1:par.ps
        for j =1:par.ps
            k    =  k+1;
            im_out(r-1+i,c-1+j)  =  im_out(r-1+i,c-1+j) + reshape( X_hat(k,:)', [par.maxr par.maxc]);
            im_wei(r-1+i,c-1+j)  =  im_wei(r-1+i,c-1+j) + reshape( W(k,:)', [par.maxr par.maxc]);
        end
    end
    im_out  =  im_out./(im_wei+eps);
    % calculate the PSNR
    PSNR =   csnr( im_out*255, par.I*255, 0, 0 );
    SSIM      =  cal_ssim( im_out*255, par.I*255, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n',ite, PSNR,SSIM);
    par.PSNR(ite,par.image) = PSNR;
    par.SSIM(ite,par.image) = SSIM;
end














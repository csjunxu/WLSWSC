function  [im_out,par]    =   BatchDenoising(par)
im_in = par.nim;
im_out    =   par.nim;
par.nSig0 = par.nSig;
% parameters for noisy image
[h,  w, ch]      =  size(im_out);
par.h = h;
par.w = w;
par.ch = ch;
par = SearchNeighborIndex( par );
for ite  =  1 : par.outerIter
    % iterative regularization
    im_out = im_out+par.delta*(par.nim - im_out);
    % image to patches and estimate local noise variance
    [Y, Sigma] = Image2Patch( im_out, im_in, par);
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par);
        if ite == 1
            Sigma = par.nSig0 * ones(size(Sigma));
        end
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2, par.maxr*par.maxc, 'single');
    W_hat = zeros(par.ps2, par.maxr*par.maxc, 'single');
    for i = 1:par.lenrc
        nlY = Y( : , blk_arr(:,i) );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        % initialize Wei for least square
        Wls = Sigma(blk_arr(:, i));
        % initialize D and S
        [Dini, Sini, ~] = svd(cov(nDCnlY'), 'econ');
        Sini = diag(Sini);
        % initialize weight for sparse coding
        Wsc = bsxfun(@rdivide, par.lambda * Sigma(blk_arr(:,i)) .^ 2, sqrt(Sini) + eps );
        % initialize C by soft thresholding
        B = Dini' * nDCnlY;
        C = sign(B) .* max(abs(B) - Wsc, 0);
        % Recovered Estimated Patches
        nDCnlYhat = WLSWSC(nDCnlY, Wls, C, par);
        nlYhat = bsxfun(@plus, nDCnlYhat, DC);
        % add DC components and aggregation
        Y_hat(:, blk_arr(:, i)) = Y_hat(:, blk_arr(:, i)) + nlYhat;
        W_hat(:, blk_arr(:, i)) = W_hat(:, blk_arr(:, i)) + ones(par.ps^2, par.nlsp);
    end
    % Reconstruction
    im_out = PGs2Image(Y_hat, W_hat, par);
    % calculate the PSNR
    PSNR =   csnr( im_out * 255, par.I * 255, 0, 0 );
    SSIM      =  cal_ssim( im_out * 255, par.I * 255, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n', ite, PSNR, SSIM);
    par.PSNR(ite, par.image) = PSNR;
    par.SSIM(ite, par.image) = SSIM;
end
return;


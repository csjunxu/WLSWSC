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
    [X, Sigma] = Image2Patch( im_out, im_in, par);
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        [nDCnlX, blk_arr, DC] = Block_Matching( X, par);
        if ite == 1
            Sigma = par.nSig0 * ones(size(Sigma));
        end
    end
    % Weighted Sparse Coding
    X_hat = zeros(par.ps2, par.maxrc, 'single');
    W_hat = zeros(par.ps2, par.maxrc, 'single');
    for i = 1:par.lenrc
        Y         =   nDCnlX(:, (i-1)*par.nlsp+1:i*par.nlsp);
        % initialize Wei for least square
        Wls = Sigma(blk_arr(:, i));
        % Recovered Estimated Patches by weighted least square and weighted
        % sparse coding model
        Yhat = WLSWSC(Y, Wls, par);
        % add DC components and aggregation
        X_hat(:, blk_arr(:, i)) = X_hat(:, blk_arr(:, i)) + bsxfun(@plus, Yhat, DC(:, i));
        W_hat(:, blk_arr(:, i)) = W_hat(:, blk_arr(:, i)) + ones(par.ps^2, par.nlsp);
    end
    % Reconstruction
    im_out = PGs2Image(X_hat, W_hat, par);
    % calculate the PSNR
    PSNR =   csnr( im_out * 255, par.I * 255, 0, 0 );
    SSIM      =  cal_ssim( im_out * 255, par.I * 255, 0, 0 );
    fprintf('Iter %d : PSNR = %2.4f, SSIM = %2.4f\n', ite, PSNR, SSIM);
    par.PSNR(ite, par.image) = PSNR;
    par.SSIM(ite, par.image) = SSIM;
end
return;


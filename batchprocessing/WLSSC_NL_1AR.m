function  [im_out,par]    =   WLSSC_NL_1AR(par)
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
    im_out = im_out + par.delta * (par.nim - im_out);
    % image to patches and estimate local noise variance
    [Y, Sigma] = Image2Patch( im_out, im_in, par);
    % estimation of noise variance
    if mod(ite-1,par.innerIter)==0
        par.nlsp = par.nlsp - 10;
        % searching  non-local patches
        blk_arr = Block_Matching( Y, par);
        if ite == 1
            Wls = par.nSig0 * ones(size(Sigma));
        end
    end
    % Weighted Sparse Coding
    Y_hat = zeros(par.ps2ch, par.maxrc, 'single');
    W_hat = zeros(par.ps2ch, par.maxrc, 'single');
    for i = 1:par.lenrc
        index = blk_arr(:, i);
        nlY = Y( : , index );
        DC = mean(nlY, 2);
        nDCnlY = bsxfun(@minus, nlY, DC);
        % Recovered Estimated Patches by weighted least square and weighted
        % sparse coding model
        nDCnlYhat = WLSSC(nDCnlY, Wls(index), par);
%         nDCnlYhat = WLSSC(nDCnlY, Wls(index), Sigma(index(1)), par);
        % update weight for least square
        Wls(index) = exp( - par.lambdals1 * sqrt(sum((nDCnlY - nDCnlYhat) .^2, 1)) ) .* exp( - par.lambdals2 * sqrt(sum(( repmat(nDCnlYhat(:, 1), [1, length(index)]) - nDCnlYhat ).^2, 1)) );
        % add DC components
        nlYhat = bsxfun(@plus, nDCnlYhat, DC);
        % aggregation
        Y_hat(:, index) = Y_hat(:, index) + nlYhat;
        W_hat(:, index) = W_hat(:, index) + ones(par.ps2ch, par.nlsp);
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


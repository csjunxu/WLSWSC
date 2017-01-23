function       [X, Sigma] = Image2Patch( im_out, im_in, par)
% record the non-local patch set and the index of each patch in
% of seed patches in image
im_out         =  single(im_out);
X          =  zeros(par.ps2, par.maxr*par.maxc, 'single');
NX          =  zeros(par.ps2, par.maxr*par.maxc, 'single');
k    =  0;
for i  = 1:par.ps
    for j  = 1:par.ps
        k    =  k+1;
        blk  = im_out(i:end-par.ps+i,j:end-par.ps+j);
        nblk  = im_in(i:end-par.ps+i,j:end-par.ps+j);
        X(k,:) = blk(:)';
        NX(k,:) = nblk(:)';
    end
end
Sigma = sqrt(abs(repmat(par.nSig0^2, 1, size(X,2)) - mean((NX - X).^2))); %Estimated Local Noise Level

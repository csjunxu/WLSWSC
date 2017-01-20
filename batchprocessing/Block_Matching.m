function       [nDCnlX, blk_arr, DC] = Block_Matching( X, par)
% record the indexs of patches similar to the seed patch
blk_arr   =  zeros(par.nlsp, par.lenrc, 'single');
% non-local patch sets of X
DC = zeros(par.ps2, par.lenrc, 'single');
nDCnlX = zeros(par.ps2, par.lenrc*par.nlsp, 'single');

for  i  =  1 : par.lenrc
    seed = X(:, par.SelfIndex(i));
    neighbor = X(:, par.NeighborIndex(1:par.NumIndex(i), i));
    dis = sum(bsxfun(@minus, neighbor, seed).^2, 1);
    [~,ind]   =  sort(dis);
    indc        =  par.NeighborIndex( ind( 1:par.nlsp ), i );
    indc(indc == par.SelfIndex(i)) = indc(1); % added on 08/01/2017
    indc(1) = par.SelfIndex(i); % to make sure the first one of indc equals to off
    blk_arr(:, i) = indc;
    temp = X( : , indc );
    DC(:, i) = mean(temp, 2);
    nDCnlX(:, (i-1) * par.nlsp+1:i * par.nlsp) = bsxfun(@minus, temp, DC(:,i));
end

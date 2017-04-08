function       blk_arr = Block_Matching( X, par)
% record the indexs of patches similar to the seed patch
blk_arr   =  zeros(par.nlsp, par.lenrc, 'single');
for  i  =  1 : par.lenrc
    seed = X(:, par.SelfIndex(i));
    neighbor = X(:, par.NeighborIndex(1:par.NumIndex(i), i));
    Dist = sum(bsxfun(@minus, neighbor, seed).^2, 1);
    [~,index]   =  sort(Dist);
    blk_arr(:,i)        =  par.NeighborIndex( index( 1:par.nlsp ), i );
    %% for real noisy images
    %     indc        =  par.NeighborIndex( index( 1:par.nlsp ), i );
    %     indc(indc == par.SelfIndex(i)) = indc(1); % added on 08/01/2017
    %     indc(1) = par.SelfIndex(i); % to make sure the first one of indc equals to off
    %     blk_arr(:, i) = indc;
end

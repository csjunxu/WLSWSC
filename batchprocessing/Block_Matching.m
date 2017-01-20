function       [nDCnlX, blk_arr, DC, par] = Block_Matching( X, par)
% record the indexs of patches similar to the seed patch
blk_arr   =  zeros(par.nlsp, par.lenr*par.lenc ,'single');
% non-local patch sets of X
DC = zeros(par.ps^2,par.lenr*par.lenc,'single');
nDCnlX = zeros(par.ps^2,par.lenr*par.lenc*par.nlsp,'single');
for  i  =  1 :par.lenr
    for  j  =  1 : par.lenc
        row = par.r(i);
        col = par.c(j);
        off = (col-1)*par.maxr + row;
        off1 = (j-1)*par.lenr + i;
        % the range indexes of the window for searching the similar patches
        rmin    =   max( row-par.Win, 1 );
        rmax    =   min( row+par.Win, par.maxr );
        cmin    =   max( col-par.Win, 1 );
        cmax    =   min( col+par.Win, par.maxc );
        idx     =   par.Index(rmin:rmax, cmin:cmax);
        idx     =   idx(:);
        neighbor       =   X(:,idx); % the patches around the seed in X
        seed       =   X(:,off);
        dis = sum(bsxfun(@minus,neighbor, seed).^2,1);
        [~,ind]   =  sort(dis);
        indc        =  idx( ind( 1:par.nlsp ) );
        indc(indc==off) = indc(1); % added on 08/01/2017
        indc(1) = off; % to make sure the first one of indc equals to off
        blk_arr(:,off1)  =  indc;
        temp = X( : , indc );
        DC(:,off1) = mean(temp,2);
        nDCnlX(:,(off1-1)*par.nlsp+1:off1*par.nlsp) = bsxfun(@minus,temp,DC(:,off1));
    end
end
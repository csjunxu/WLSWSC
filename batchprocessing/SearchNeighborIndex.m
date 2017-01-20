function  [NeighborIndex, NumIndex, SelfIndex]  =  SearchNeighborIndex(par)
% This Function Precompute the all the patch indexes in the Searching window
% -Neighbor_arr is the array of neighbor patch indexes for each keypatch
% -Num_arr is array of the effective neighbor patch numbers for each keypatch
% -SelfIndex_arr is the index of keypatches in the total patch index array
par.maxr = h-par.ps+1;
par.maxc = w-par.ps+1;
r          =  1:par.step:par.maxr;
par.r          =  [r r(end)+1:par.maxr];
c          =  1:par.step:par.maxc;
par.c          =  [c c(end)+1:par.maxc];
par.lenr = length(par.r);
par.lenc = length(par.c);
par.ps2 = par.ps^2;
par.maxrc = par.maxr*par.maxc;
par.lenrc = par.lenr*par.lenc;
% index of each patch in image
par.Index     =   (1:par.maxr*par.maxc);
par.Index    =   reshape(par.Index,par.maxr,par.maxc);

NeighborIndex    =   int32(zeros(4 * par.Win^2, par.lenr * par.lenc));
NumIndex        =   int32(zeros(1, par.lenr * par.lenc));
SelfIndex   =   int32(zeros(1, par.lenr * par.lenc));

for  i  =  1 : par.lenr
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
        
        NumIndex(off1)  =  length(idx);
        NeighborIndex(1:NumIndex(off1),off1)  =  idx;
        SelfIndex(off1) = off;
    end
end
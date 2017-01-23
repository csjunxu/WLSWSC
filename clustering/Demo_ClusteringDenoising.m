clear;
Original_image_dir  =    'C:\Users\csjunxu\Desktop\PGPD_TIP\20images\';
fpath = fullfile(Original_image_dir, '*.png');
im_dir  = dir(fpath);
im_num = length(im_dir);

nSig = 40;
par.ps = 7; % patch size
nlsp = 10;
par.step = 2; % the step of two neighbor patches
par.changeD = 3;
par.Win = min(3*par.ps, 20);
par.nlsp = nlsp;  % number of non-local patches

par.IteNum = 3*par.changeD;
for cls_num= 32
    par.cls_num = cls_num; % number of clusters
    for delta = 0.08
        par.delta = delta;
        for lambda = 0.17:0.02:0.23
            par.lambda=lambda;
            % record all the results in each iteration
            par.PSNR = zeros(par.IteNum,im_num,'single');
            par.SSIM = zeros(par.IteNum,im_num,'single');
            T512 = [];
            T256 = [];
            for i = 1:im_num
                par.maxiter = 20;% number of iterations in PG-GMM training
                par.image = i;
                par.nSig = nSig/255;
                par.I =  single( imread(fullfile(Original_image_dir, im_dir(i).name)) )/255;
                %                 S = regexp(im_dir(i).name, '\.', 'split');
                randn('seed',0);
                par.nim =   par.I + par.nSig*randn(size(par.I));
                %
                fprintf('%s :\n',im_dir(i).name);
                PSNR =   csnr( par.nim*255, par.I*255, 0, 0 );
                SSIM      =  cal_ssim( par.nim*255, par.I*255, 0, 0 );
                fprintf('The initial value of PSNR = %2.4f, SSIM = %2.4f \n', PSNR,SSIM);
                %
                time0 = clock;
                [im_out,par]  =  Denoising(par);
                if size(par.I,1) == 512
                    T512 = [T512 etime(clock,time0)];
                    fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                elseif size(par.I,1) ==256
                    T256 = [T256 etime(clock,time0)];
                    fprintf('Total elapsed time = %f s\n', (etime(clock,time0)) );
                end
                im_out(im_out>1)=1;
                im_out(im_out<0)=0;
                % calculate the PSNR
                par.PSNR(par.IteNum,par.image)  =   csnr( im_out*255, par.I*255, 0, 0 );
                par.SSIM(par.IteNum,par.image)      =  cal_ssim( im_out*255, par.I*255, 0, 0 );
%                 imname = sprintf('nSig%d_clsnum%d_delta%2.2f_lambda%2.2f_%s', nSig, cls_num, delta, lambda, im_dir(i).name);
%                 imwrite(im_out,imname);
%                 fprintf('%s : PSNR = %2.4f, SSIM = %2.4f \n',im_dir(i).name, par.PSNR(par.IteNum,par.image),par.SSIM(par.IteNum,par.image)     );
            end
            mPSNR=mean(par.PSNR,2);
            [~, idx] = max(mPSNR);
            PSNR =par.PSNR(idx,:);
            SSIM = par.SSIM(idx,:);
            mSSIM=mean(SSIM,2);
            mT512 = mean(T512);
            sT512 = std(T512);
            mT256 = mean(T256);
            sT256 = std(T256);
            fprintf('The best PSNR result is at %d iteration. \n',idx);
            fprintf('The average PSNR = %2.4f, SSIM = %2.4f. \n', mPSNR(idx),mSSIM);
            name = sprintf('C:/Users/csjunxu/Desktop/PGPD_TIP/WLSWSC/CP_nSig%d_clsnum%d_delta%2.2f_lambda%2.2f_.mat',nSig, cls_num, delta, lambda);
            save(name,'nSig','PSNR','SSIM','mPSNR','mSSIM','mT512','sT512','mT256','sT256');
        end
    end
end
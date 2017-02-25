% single weighted: weighted least square and sparse coding framework
function  X = WLSSC_DCW(Y, Sigma, Wls, par)
% initialize D
YW = bsxfun(@times, Y, Wls);
[U, ~, V] = svd(YW * YW', 'econ');
D = V * U';
f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update W for weighted sparse coding
    Wsc = par.lambdasc(par.ite) ./ Wls.^2;
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - repmat(Wsc, [size(B, 1), 1]), 0);
    % update D
    if par.model == 1
        % model 1
        CW = bsxfun(@times, C, Wls);
        [U, ~, V] = svd( CW * Y', 'econ');
    else
        % model 2
        CW = bsxfun(@times, C, Wls);
        YW = bsxfun(@times, Y, Wls);
        [U, ~, V] = svd( CW * YW', 'econ');
    end
    D = V * U';
    % update weight for least square
    res = sqrt(sum((Y - D * C) .^2, 1) / size(Y, 1));
    Wls = 1 ./ Sigma .* mean(res) ./ res;
    % energy function
    DT = bsxfun(@times, Y - D * C, Wls);
    DT = norm(DT, 'fro');
    %     DT = DT(:)'*DT(:);
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT ^ 2 + par.lambdasc * RT;
%     fprintf('WLSSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;

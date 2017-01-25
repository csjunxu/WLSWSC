% double weighted: weighted least square and weighted sparse coding framework
function  X = WLSWSC(Y, Wls, par)
% initialize D and S
YW = bsxfun(@times, Y, Wls);
[U, S, V] = svd(YW * Y', 'econ');
D = U * V';
%     D = U;
S = diag(S);
f_curr = 0;
% C = zeros(size(D, 1), size(Y, 2));
for i=1:par.WWIter
    f_prev = f_curr;
    %     C_prev = C;
    % update W for weighted sparse coding
    %     Wsc = bsxfun(@rdivide, par.lambdasc * Wls .^ 2, sqrt(S) + eps );
    % update C by soft thresholding
    B = D' * Y;
    %     C = sign(B) .* max(abs(B) - Wsc, 0);
    C = sign(B) .* max(abs(B) - par.lambdasc, 0);
    % update D and S
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
    %     S = diag(S);
    
    %     residual = norm(C - C_prev, 1);
    %     fprintf('C residual, %d th: %2.8f\n', i, residual);
    %     if residual < par.epsilon
    %         %     if (abs(f_prev - f_curr) / f_curr < par.epsilon)
    %         break;
    %     end
    %     DT = bsxfun(@times, Y - D * C, Wls);
    DT = Y - D * C;
    DT = DT(:)'*DT(:);
    %     RT = Wsc .*  C;
    RT = norm(C, 1);
    f_curr = 0.5 * DT + par.lambdasc * RT;
    fprintf('WLSWSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;

% double weighted: weighted least square and weighted sparse coding framework
function  X = LSSC(Y, par)
% initialize D and S
[U, S, V] = svd(Y * Y', 'econ');
D = U * V';
% S = diag(S);
f_curr = 0;
for i=1:par.WWIter
    f_prev = f_curr;
    % update W for weighted sparse coding
    %     Wsc = bsxfun(@rdivide, par.lambdasc * Wls .^ 2, sqrt(S) + eps );
    % update C by soft thresholding
    B = D' * Y;
    %     C = sign(B) .* max(abs(B) - Wsc, 0);
    C = sign(B) .* max(abs(B) - par.lambdasc, 0);
    % update D and S
    [U, ~, V] = svd( C * Y', 'econ');
    D = U * V';
    %     S = diag(S);
    
    %     residual = norm(C - C_prev, 1);
    %     fprintf('C residual, %d th: %2.8f\n', i, residual);
    %     if residual < par.epsilon
    %         %     if (abs(f_prev - f_curr) / f_curr < par.epsilon)
    %         break;
    %     end
    %     DT = bsxfun(@times, Y - D * C, Wls);
    DT = Y - D * C;
    DT = norm(DT, 'fro');
    %     RT = Wsc .*  C;
    RT = sum(sum(abs(C)));
    f_curr = 0.5 * DT^2 + par.lambdasc * RT;
    fprintf('WLSWSC Energy, %d th: %2.8f\n', i, f_curr);
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
end
% update X
X = D * C;
return;

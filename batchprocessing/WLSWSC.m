% double weighted: weighted least square and weighted sparse coding framework
function  X = WLSWSC(Y, Wls, par)
X = Y;
% initialize D and S
[D, S, ~] = svd(full(Y), 'econ');
S = diag(S);
% % update W for weighted sparse coding
% Wsc = bsxfun(@rdivide, par.lambdasc * Wls .^ 2, sqrt(S) + eps );
f = 0;
for i=1:par.WWIter
    f_prev = f;
    % update W for weighted sparse coding
    Wsc = bsxfun(@rdivide, par.lambdasc * Wls .^ 2, sqrt(S) + eps );
    % update C by soft thresholding
    B = D' * Y;
    C = sign(B) .* max(abs(B) - Wsc, 0);
    % update D and S
    CW = bsxfun(@times, C, Wls);
    YW = bsxfun(@times, X, Wls);
    [U, S, V] = svd( CW * YW', 'econ');
    D = U * V';
    %     D = U;
    S = diag(S);
    
    DT = bsxfun(@times, Y - D * C, Wls);
    DT = DT(:)'*DT(:) / 2;
    RT = Wsc .*  C;
    RT = norm(RT, 1);
    f_curr = DT + RT;
    if (abs(f_prev - f_curr) / f_curr < par.epsilon)
        break;
    end
    fprintf('WLSWSC Energy: %2.4f\n', f_curr);
end
% update X
X = D * C;
return;

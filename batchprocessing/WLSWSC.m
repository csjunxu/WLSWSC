% weighted least square and weighted sparse coding
function  X = WLSWSC(Y, Wei, C, par)
X = Y;
% update Wei
% IS NOT NEED
for i=1:par.WWIter
    % update D and S
    CW = bsxfun(@times, C, Wei);
    YW = bsxfun(@times, X, Wei);
    [D, S, ~] = svd( CW * YW', 'econ');
    S = diag(S);
    % update
    Wsc = bsxfun(@rdivide, par.lambda * (Wei .^ 2), sqrt(S) + eps );
    % update C
    B = D' * Y;
    C = sign(B) .* max(abs(B) - Wsc, 0);
    % update X
    X = D * C;
end
return;

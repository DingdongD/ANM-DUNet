function T = toeplitz_adjoint(A)
    N = size(A,1);
    T = zeros(N,1);
    T(1) = sum(diag(A));
    for n = 1:(N-1)
        T(n + 1) = sum(diag(A,n));
    end
end
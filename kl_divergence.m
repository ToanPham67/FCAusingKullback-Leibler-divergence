function kl_div = kl_divergence(p, q, a, b)
    % Compute the Kullback-Leibler Divergence between two continuous probability density functions.
 % Define the integrand for KL divergence calculation
    integrand = @(x) p(x) .* log((p(x) + eps) ./ (q(x) + eps));

    % Compute KL divergence using numerical integration
    kl_div = integral(integrand, a, b);
end

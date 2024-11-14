function [kl_div,integrand_values] = kl_divergence_plot(p, q, a, b)
    % Compute the Kullback-Leibler Divergence between two continuous probability density functions.

    % Define the integrand for KL divergence calculation
    integrand = @(x) p(x) .* log((p(x) + eps) ./ (q(x) + eps));

    % Compute KL divergence using numerical integration
    kl_div = integral(integrand, a, b);

    % Generate x values for plotting
    x_values = linspace(a, b, 1000);

    % Compute the integrand values
    integrand_values = integrand(x_values);

%     % Plot the integrand
%     figure;
%     plot(x_values, integrand_values, 'b', 'LineWidth', 2);
%     xlabel('x');
%     ylabel('Integrand');
%     title('Integrand of KL Divergence');
%     grid on;
% 
%     % Shade the area under the curve
%     patch([x_values, fliplr(x_values)], [integrand_values, zeros(size(integrand_values))], 'c', 'EdgeColor', 'none', 'FaceAlpha', 0.4);
% 
%     % Display KL divergence on the plot
%     text(0.5, 0.5, ['KL Divergence: ', num2str(kl_div)], 'Units', 'normalized', 'FontSize', 12, 'BackgroundColor', [1 1 1]);
% 
%     % Return KL divergence
%     disp(['KL Divergence:', num2str(kl_div)]);
end

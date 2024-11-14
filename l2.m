function l2_distance = l2(P, Q)
    % Compute the L1 distance between two discrete probability distributions.

    % Ensure that P and Q are probability distributions (sum to 1)
    P = P / sum(P);
    Q = Q / sum(Q);

    % Compute L1 distance
    l2_distance = sum(abs(P - Q).^2);
end

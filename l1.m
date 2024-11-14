function l1_distance = l1(P, Q)
    % Compute the L1 distance between two discrete probability distributions.

    % Ensure that P and Q are probability distributions (sum to 1)
    P = P / sum(P);
    Q = Q / sum(Q);

    % Compute L1 distance
    l1_distance = sum(abs(P - Q));
end

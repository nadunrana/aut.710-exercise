% Create workspace
x_min = 0;
x_max = 10;
x = linspace(x_min, x_max, 100);

% Position belief distribution
pd_x = makedist('Normal', 'mu', 4, 'sigma', 0.25);
p_x = pdf(pd_x, x);
eta = sum(p_x);
p_x = 1/eta * p_x;

% Plot Position Belief with out D
figure("Name", "Part 1")
subplot(2, 1, 1);
bar(x, p_x);
title("Position Belief without D");

% Calculate mean and covariance
Ex = expected_value(x, p_x);
COVx = covariance(x, p_x);

% Sensor noise distribution
pd_noise = makedist('Normal', 'mu', 0, 'sigma', 0.15); % sensor noise model
z_sensor = 3.6; % measurement value

% Position belief with sensor measurement distribution
pz_x = pdf(pd_noise, z_sensor - x);
px_z = pz_x.*p_x;
eta = sum(px_z);
px_z = 1/eta * px_z;

% Plot Position belief with D
subplot(2, 1, 2);
bar(x, px_z);
title("Position Belief with D = 3.6");

% Prepare figure for part 2
figure("Name", "Part 2");

% Environment noise distribution
pd_w = makedist('Normal', 'mu', 0, 'sigma', 0.2);

% Step calculating loop
p_xpr = p_x;
for i = 1:3
    u = -0.1 * expected_value(x, p_x);
    for k = 1:100
        p_motion = pdf(pd_w, x(k) - x - u);
        p_xpr(k) = sum(p_motion.*p_x);
    end
    eta = sum(p_xpr);
    p_xpr = 1/eta * p_xpr;

    subplot(3, 1, i);
    title([num2str(i), "th iteration"])
    bar(x, p_xpr);
    expected_value(x, p_x);
    p_x = p_xpr;
end

% Position belief with sensor measurement distribution
z_sensor = 2.5;
pz_x = pdf(pd_noise, z_sensor - x);
px_z = pz_x.*p_x;
eta = sum(px_z);
px_z = 1/eta * px_z;

% Plot Position belief with D
figure("Name", "Part 3");
bar(x, px_z);
title("Position Belief with D = 2.5");


function e_x = expected_value(x, p)
    e_x = sum(x.*p);
end

function cov_x = covariance(x, p)
    Ex = expected_value(x, p);
    cov_x = expected_value((x - Ex).^2, p);
end

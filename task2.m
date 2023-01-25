% Set 1
x = [-10:.01:10];
y1 = normpdf(x, 0, 0.1);
y1 = 1/sum(y1) * y1;
y2 = normpdf(x, 0, 0.3);
y2 = 1/sum(y2) * y2;
y3 = normpdf(x, 0, 0.7);
y3 = 1/sum(y3) * y3;

figure("Name", "Part 1");
subplot(2, 1, 1);
title("Set 1");
hold on;
plot(x, y1);
plot(x, y2);
plot(x, y3);

% Set 2
y4 = normpdf(x, -5, 0.5);
y4 = 1/sum(y4) * y4;
y5 = normpdf(x, 0, 0.5);
y5 = 1/sum(y5) * y5;
y6 = normpdf(x, 5, 0.5);
y6 = 1/sum(y6) * y6;

subplot(2, 1, 2);
title("Set 2");
hold on;
plot(x, y4);
plot(x, y5);
plot(x, y6);

% Mean and Covariance
T = importdata('AUT710_ex1_task2.csv');

expected_value = mean(T);
standard_deviation = std(T);
[hc,edges] = histcounts(T, 100);
hc = 1/sum(hc) * hc;
centres = edges(1:length(edges)-1) + mean(diff(edges))/2;
x = linspace(min(centres), max(centres));  
n = normpdf(x, expected_value, standard_deviation);

figure("Name", "Part 2");
bar(ctrs, hc);
title("Probability vs Random Variable");

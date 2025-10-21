bends = [500, 1000, 2500, 5000, 7500, 10000];
% bends = [500];

vel = zeros(size(bends));

i=1;
for b = bends
    subplot(2,3,i);
    fname = strcat('../save/', num2str(b), '/U_rigid_mean.txt');
    dat = readmatrix(fname);

    x_velocity = zeros(length(dat), 1);

    x_velocity(:, 1) = dat(:,2);
    plot(x_velocity);
    title(strcat('k=', num2str(b)))

    if i == 1 || i == 4
        ylabel('x velocity');
    end
    if i >= 4
        xlabel('vaguely time')
    end
    i = i + 1;
end
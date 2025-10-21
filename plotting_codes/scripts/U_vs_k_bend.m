clear all
close all

bends = [500, 1000, 2500, 5000, 7500, 10000];
vel = zeros(size(bends));
deflection = zeros(size(bends));

eta = 1.0;
freq = 3;
R = 1.0;

i = 1;
for b = bends
    fname = strcat('../save/', num2str(b), '/U_rigid_mean.txt');
    dat = readmatrix(fname);

    v = dat(end, 2);
    vel(i) = abs(v);

    deflection(i) = dat(end, 8);
    i = i + 1;
end

vel = vel ./ vel(end);
bends = bends ./ ( (8*pi*eta*(R^3)) * (2*pi*freq) );

% Main plot
lw=10;
ms=25;

figure()
red = '#ff8c00';
plot(bends, vel, 'color', red, linewidth=lw, marker='square', markersize=ms, markerfacecolor=red);
ylabel('$u_x / u_{flat}$', 'interpreter','latex');
xlabel('$k_b / 8\pi\eta R^3 (2\pi f)$');

xlim([0.5,21.5])
xticks([1,5,10,15,20]);
daspect([20,0.4,1])

% Inset axes
inset_pos = [0.35, 0.3, 0.4, 0.4]; % [x, y, width, height] in normalized units
yellow = '#ba2329';
inset_ax = axes('Position', inset_pos);
plot(inset_ax, bends, deflection, 'Color',yellow, LineWidth=lw, Marker='diamond', MarkerSize=ms, MarkerFaceColor=yellow);
ylabel(inset_ax, 'Total deflection / R');

xlim([0.5,21.5])
xticks([1,5,10,15,20]);
daspect([20,4,1])

% xlabel(inset_ax, 'bending stiffness');
box(inset_ax, 'on');


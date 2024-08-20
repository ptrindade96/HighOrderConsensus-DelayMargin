clear
clc

%% Prepare the example

% Problem data
bar_tau_d = 0.1;                    % Prescribed delay-margin
tau = 0.99*bar_tau_d;                % Delay in the example 
M = 11;                             % Order
L = [    1 -1  0  0  0  0;          % Laplacian
         0  2 -1 -1  0  0;
         0  0  1  0 -1  0;
        -1  0  0  1  0  0;
         0 -1  0 -1  2  0;
         0  0 -1  0  0  1   ];

% Some preliminary computations
n = length(L(1,:));
lambda = eig(L);
lambda = lambda(abs(lambda)>1e-8);   % Remove the null eigenvalue

%% Design the coupling gains.

% Step 1 - Choose the zeros
LL = 5;      % Number of open-loop zeros with damping less that sqrt(2)/2
wn_min = 2.5;
wn_max = 3.5;
wn_h = abs(wn_min-wn_max)/floor((M-1)/2-1);
wn = wn_min:wn_h:wn_max;
z0 = -1+wn*1i;
if rem(M-1,2)==0
    z0 = [conj(z0),z0];
else
    z0 = [conj(z0),-1,z0];
end
z0 = z0/abs(sum(real(z0)));

% Step 2 - Compute the sets Omega_phi 
lambda_aux = lambda(imag(lambda)>-1e-8);
psi_aux = phase(lambda_aux);
Omega_phi = cell([length(lambda_aux),1]);
opt = optimoptions('fsolve','Display','none');
epsilon = 1e-8;
for l=1:length(lambda_aux)
    ks = -1-floor((M-2*(1-psi_aux(l)/pi))/4-epsilon):floor((M-2*(1+psi_aux(l)/pi))/4-epsilon);
    fun = @(w) sum(angle(1i*w-z0'),1)-sign(w)*M*pi/2+psi_aux(l)*ones(1,length(ks))+(2*ks+1)*pi;
    Omega_phi{l} = fsolve(fun,ks+0.5,opt);
end

% Step 3 - Compute the smallest gain for stability, and select a gain
K_min = 0;
for l=1:length(lambda_aux)
    K_min = max(K_min,max(abs(Omega_phi{l}).^M./prod(abs(1i*Omega_phi{l}-z0'),1))/abs(lambda_aux(l)));
end
K0 = 4;

%%% Step 4 - Compute the gain crossover frequencies
% 4.1 - Construct the polynomial that defines the gain crossover frequencies.
coef = 1;  
z_r = z0(abs(imag(z0))<1e-15);    % Real zeros
z_c = z0(imag(z0)>1e-15);         % Complex zeros
for i=1:length(z_r)
    coef = conv(coef,[1 z_r(i)^2]);
end
for i = 1:length(z_c)
    coef = conv(coef,[1,2*(2*real(z_c(i))^2-abs(z_c(i))^2),abs(z_c(i))^4]);
end
% 4.2 - Compute the gain crossover frequencies for each lambda
Omega_0_pm = cell([length(lambda_aux),1]);
for l = 1:length(lambda_aux)
    W0 = roots([1,-K0^2*abs(lambda_aux(l))^2*coef]);
    Omega_0 = sqrt(real(W0(abs(imag(W0))<1e-15&real(W0)>0)));
    Omega_0_pm{l} = [-Omega_0',Omega_0'];
end

% Step 5 - Compute the delay margin
bar_tau = inf;
for l=1:length(lambda_aux)
    phase_plus_pi = sum(angle(1i*Omega_0_pm{l}-z0'),1)-sign(Omega_0_pm{l})*M*pi/2 + psi_aux(l)' + pi;
    phase_margins = phase_plus_pi + sign(Omega_0_pm{l}).*ceil(-sign(Omega_0_pm{l}).*(phase_plus_pi)/(2*pi))*2*pi;
    bar_tau = min([bar_tau,phase_margins./Omega_0_pm{l}]);
end

% Step 6 - Compute the scaling factor alpha and scale the gain and zeros
alpha = bar_tau_d/bar_tau;
K = K0/alpha;
z = z0/alpha;

% Step 7 - Compute all the gains
g = 1;
for i=1:M-1
    g = conv(g,[1 -z(i)]);
end
g = real(g);

%% Simulate the dynamics

% Define the time vector
T = 600;
h = 0.5e-3;
t = 0:h:T;

% Compute the "discrete" delay
K_tau = ceil(tau/h);

% Define the initial conditions
x0 = [(1:n)';zeros((M-1)*n,1)];

% Simulate the dynamics
x = zeros(n*M,length(t));
x(:,1) = x0; 
AA = kron(eye(M)+h*diag(ones(1,M-1),1),eye(n));
KK = h*K*kron(flip(g),L);
for i=2:length(t)
    x(1:(M-1)*n,i) = x(1:(M-1)*n,i-1) + h*x(n+1:M*n,i-1);% + h^2/2*[x_(2*n+1:M*n,i-1);zeros(n,1)];
    if i > K_tau+1
        x((M-1)*n+1:M*n,i) = x((M-1)*n+1:M*n,i-1) - KK*x(:,i-1-K_tau);
    end
end

% Compute the consensus dynamics
o = ones(n,1);
r = null(L');
r = r/(r'*o);
xc = zeros(1,length(t));
for i=1:M
    xc = xc + (r'*x0((i-1)*n+1:i*n))*t.^(i-1)/factorial(i-1);
end

% Correct the consensus value
y = o*xc + (eye(n)-1*r')*x(1:n,:);
clear x xc

%% Plot the state evolution

line_width = 1;
h = figure();
sampling = floor(linspace(1,length(t),50000));
plot(t(sampling),y(:,sampling),'LineWidth',1)
xlabel("Time, $t$ [s]",Interpreter="latex")
ylim([-5,10])
l_P = legend("1","2","3","4","5","6");
l_P.Location = "southeast";
l_P.Orientation = "vertical";
l_P.ItemTokenSize = [15,9]*0.7;
l_P.NumColumns = 2;
grid on


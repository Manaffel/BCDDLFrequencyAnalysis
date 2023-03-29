%Load data
data = readmatrix('ExampleSignals.txt');
T = data(:,2);
X = data(:,5);
Y = data(:,4);
%Mean samplign time and frequency
Ts = mean(T(2:end,1)-T(1:end-1,1));
fs = 1/Ts;
%Measure FRF
[frf,f] = tfestimate(X,Y,round(length(T)/64),[],[],fs);
%Get measured magnitude and phase
magfrf = 20*log10(abs(frf));
phfrf = angle(frf)*180/pi; 

%Set options for TF fit
opt = tfestOptions;
opt.EnforceStability = true;
iodelay = NaN;

%Fit TF
npols = 16;
nzeros = 13;
sysd = tfest(X,Y,npols,nzeros,iodelay,'Ts',Ts,opt);
sysc = d2c(sysd)

%Get fitted magnitude and phase
[magfit,phfit] = bode(sysc,2*pi*f);
magfit = 20*log10(squeeze(magfit));
phfit = (squeeze(phfit));

%Plot data
tiledlayout(2,1)
ax1 = nexttile;
semilogx(ax1,f,magfrf)
ylabel('Magnitude [dB]')
grid on
hold on
semilogx(ax1,f,magfit)

hold off
ax2 = nexttile;
semilogx(ax2,f,phfrf)
xlabel('Frequency [Hz]')
ylabel('Phase [°]')
grid on
hold on
semilogx(ax2,f,phfit)
hold off 
legend(ax1,'FRF','TF Fit')

%Save Numerator and Denominator as txt
writematrix(sysc.Numerator,'TFNumerator.txt','Delimiter',' '); 
writematrix(sysc.Denominator,'TFDenominator.txt','Delimiter',' ');
%% Reaktor i GKT-Projekt
% etan <-> eten + H2
% A <-> B + H2
global R K1 P dHr0 dA dB dC dD Tref T0 FH_2O %Gör till globalvariabler för att slippa definiera de i funktionen igen 
R=8.3145;                                                         %[J/mol*K]                               
P=1;                                                              %[bar]
T0=950;                                                           %[K]
K1=5.042e4;                                                       %[konstant]
FA0=113;                                                          %[mol/s]
FB0=0;                                                            %[mol/s]
FH20=0;                                                           %[mol/s]
FH_2O=1130;                                                       %[mol/s]
W=[0 2500];                                                       %[kg]
W1=[W(end) 5000];                                                 %[kg]
Tref=298;                                                         %[K]

%Reaktionsentalpin
dHfA=-83.82*10^3;                                                 %[J/mol]
dHfB=52.28*10^3;                                                  %[J/mol]
dHfH2=0;                                                          %[J/mol]
dHr0=dHfB+dHfH2-dHfA;

%  ethylene   +H2             -etan
dA=+3.806      +27.14        - 5.409;                             %[Null]                              
dB=+0.1566     +0.009274     - 0.1781;                            %[Null]
dC=-8.324e-5   -1.381*10^-5  + 6.938e-5;                          %[Null]
dD=+1.755e-8   + 7.645*10^-9 - 8.713e-9;                          %[Null]

%ODE över reaktor 1 med initialvärden från ovan
[W,F]=ode15s(@odeeqq,W,[FA0 FB0 FH20 FH_2O T0]);

%ODE över reaktor 2 med initialvärden definerade utifrån resultat av första ODE:n
[W1,F1]=ode15s(@odeeqq,W1,[F(end,1) F(end,2) F(end,3) F(end,4) T0]);

%Konkatenerar F1 till F för att räkna ut omsättningen.
Ftot=vertcat(F(:,1),F1(:,1));
Conv=(FA0-Ftot)./FA0;

%Konkatenerar W1 till W för att matcha vektorlängden till plot.
Wtot=vertcat(W,W1);

%Subplotar
subplot(2,3,1),plot(W,F(:,1),'r',W1,F1(:,1),'r'), title('A') ,xlabel('kg katalysator'), ylabel('molflöde')
subplot(2,3,2),plot(W,F(:,2),'b',W1,F1(:,2),'b'),title('B'),xlabel('kg katalysator'), ylabel('molflöde')
subplot(2,3,3),plot(W,F(:,3),'g',W1,F1(:,3),'g'),title('H2'),xlabel('kg katalysator'), ylabel('molflöde')
subplot(2,3,4),plot(W,F(:,5),'k',W1,F1(:,5),'k'),title('T'),xlabel('kg katalysator'), ylabel('temperatur')
subplot(2,3,5),plot(Wtot,Conv,'m'),title('X_A'),xlabel('kg katalysator'), ylabel('Conversion')

%% Ugn innan reaktorblock
% Molmassor
MH20=1.0079*2+16;
MH2= 1.0079*2;
MC2H6= 30.07;
MC2H4=28.05;

% Molflöden
FH20=1130;
FH2=0;
FC2H6=111.87;
FC2H4=0.565;

% Temperaturer i K
Tin=469;
Tut=950;
Tm=(Tut+Tin)/2;

% Värmekapaciteter
CPH2=27.14 + 0.009274*Tm - 1.381*10^-5*Tm^2 + 7.645*10^-9*Tm^3;    %[J/mol*K]
CPH2O=32.24 + 0.001924*Tm + 1.055*10^-5*Tm^2 - 3.596*10^-9*Tm^3;   %[J/mol*K]
CPC2H6= 3.806 + 0.1566*Tm - 8.324e-5*Tm^2 + 1.755e-8*Tm^3;      %[J/mol*K]
CPC2H4= 5.409 + 0.1781*Tm - 6.938e-5*Tm^2 + 8.713e-9*Tm^3;    %[J/mol*K]

% Totalt värmebehov 
q=(FH20*CPH2O + FH2*CPH2+ FC2H6*CPC2H6 + FC2H4*CPC2H4)*(Tut-Tin)/10^6;

disp(['Ugn innan reaktorblock'])
disp(['Ugnens värmebehov:    q = ' num2str(q)   ' MW' ])

% Kostnad för drift
H= 8000; % antalet timmar på ett år
q_kwh=q*10^3*H; % kWH per år för Ugnen

el= 0.3*q_kwh; % Driftkostnad med El
Ngas=0.2*q_kwh; % Driftkostnad med Naturgas

% Kostnad för inköp

% Cylinder
a=80000; b=109000; % Kostnadsparametrar
S=q; % Storleksparameter
n=0.8; % Utrustningsexponent

C10= a+b*S^n; % Inköpskostnad år 2010
C20= C10*9.99; % Inköpskostnad år 2020 (x10 för dollar)

Kostnad_El= (el+C20); % Totalkostnad El-ugn 2020
Kostnad_Ngas= (Ngas+C20); % Totalkostnad Naturgas-ugn 2020

disp(['Inköpskostnad för Ugn = ' num2str(C20)   ' kr' ])
disp(['Driftkostnad El-ugn: = ' num2str(el)   ' kr/år' ])
disp(['Driftkostnad Naturgas-ugn: ' num2str(Ngas)   ' kr/år' ])
disp(['---------------------------------------------------------'])

%% Ugn efter reaktor 1
% Molmassor
MH20=1.0079*2+16;
MH2= 1.0079*2;
MC2H6= 30.07;
MC2H4=28.05;

% Molflöden
FH20=1130;
FH2=39.43;
FC2H6=73.57;
FC2H4=39.43;

% Temperaturer i C
Tin=850;
Tut=950;
Tm=(Tut+Tin)/2;

% Värmekapaciteter
CPH2=27.14 + 0.009274*Tm - 1.381*10^-5*Tm^2 + 7.645*10^-9*Tm^3;    %[J/mol*K]
CPH2O=32.24 + 0.001924*Tm + 1.055*10^-5*Tm^2 - 3.596*10^-9*Tm^3;   %[J/mol*K]
CPC2H6= 3.806 + 0.1566*Tm - 8.324e-5*Tm^2 + 1.755e-8*Tm^3;      %[J/mol*K]
CPC2H4= 5.409 + 0.1781*Tm - 6.938e-5*Tm^2 + 8.713e-9*Tm^3;    %[J/mol*K]

% Totalt värmebehov 
q=(FH20*CPH2O + FH2*CPH2+ FC2H6*CPC2H6 + FC2H4*CPC2H4)*(Tut-Tin)/10^6;

disp(['Ugn efter reaktorblock'])
disp(['Ugnens värmebehov:    q = ' num2str(q)   ' MW' ])

% Kostnad för drift
H= 8000; % antalet timmar på ett år
q_kwh=q*10^3*H; % kWH per år för Ugnen

el= 0.3*q_kwh; % Driftkostnad med El
Ngas=0.2*q_kwh; % Driftkostnad med Naturgas

% Kostnad för inköp

% Cylinder
a=80000; b=109000; % Kostnadsparametrar
S=q; % Storleksparameter
n=0.8; % Utrustningsexponent

C10= a+b*S^n; % Inköpskostnad år 2010
C20= C10*9.99; % Inköpskostnad år 2020 (x10 för dollar)

Kostnad_El= (el+C20); % Totalkostnad El-ugn 2020
Kostnad_Ngas= (Ngas+C20); % Totalkostnad Naturgas-ugn 2020

disp(['Inköpskostnad för Ugn = ' num2str(C20)   ' kr' ])
disp(['Driftkostnad El-ugn: = ' num2str(el)   ' kr/år' ])
disp(['Driftkostnad Naturgas-ugn: ' num2str(Ngas)   ' kr/år' ])
disp(['---------------------------------------------------------'])

%% Bubbel- och daggpungktsberäkning och jämviktskurva för lättflyktig komponent
% 1 = eten  2 = etan
clear all;
%Wilson parameters
W12=1.10698;
W21=0.67066;
%Antoine constants for K, mmHg, log10
A1=15.5368;  B1=1347.01;  C1=-18.15;  %eten
A2=15.6637;  B2=1511.42;  C2=-17.16;  %etan
%total pressure
P=22502;  %mmHg
x1=linspace(0,1);
x2=1-x1;
%activity coefficients at x1
gamma1=exp( (-log(x1+W12.*x2)) + x2.*( W12./(x1+W12.*x2) - W21./(W21.*x1+x2) ));
gamma2=exp(-log(x2+W21.*x1) - x1.*( W12./(x1+W12.*x2) - W21./(W21.*x1+x2) ));
BP1=B1/(A1-log(P))-C1; BP2=B2/(A2-log(P))-C2;
Tstart=(BP1+BP2)/2; %temperature at which to start the search
%use fsolve function to find bubble point temperature (Tb) for x1
%find_Tb is a function we need to create that will check if a certain value of T satisfies y1+y2-1=0
%current value of x1 and other constants are passed to find_Tb
for i=1:length(x1)
Tb(i)=fsolve(@(T)find_Tb(T,x1(i),gamma1(i),gamma2(i),A1,B1,C1,A2,B2,C2,P),Tstart);

P_1_0=exp(A1-B1/(Tb(i)+C1));
P_2_0=exp(A2-B2/(Tb(i)+C2));
y1(i)=(gamma1(i)*x1(i)*P_1_0)/P;
y2(i)=(gamma2(i)*(1-x1(i))*P_2_0)/P;
end

%ploting figures of bubble- and dewpointcurves as well as equilibrium curve for ethene
figure(1)
plot(x1,Tb,y1,Tb),title('Bubbel- och Daggpunktskurva (P=30 bar)'),xlabel('x1,y1'),ylabel('T(K)'), grid on, hold on
yh=x1;
figure(2)
plot(x1,y1,x1,yh),title('Jämviktskurva för Eten (P=30 bar)'), xlabel('x1'), ylabel('y1'), grid on, hold on

%Egenskaper hos Feeden
xf=0.5;
gamma11=exp( (-log(xf+W12.*(1-xf))) + (1-xf).*( W12./(xf+W12.*(1-xf)) - W21./(W21.*xf+(1-xf)) ));
gamma22=exp(-log((1-xf)+W21.*xf) - xf.*( W12./(xf+W12.*(1-xf)) - W21./(W21.*xf+(1-xf)) ));
Tb=fsolve(@(T)find_Tb(T,xf,gamma11,gamma22,A1,B1,C1,A2,B2,C2,P),Tstart)
P10=exp(A1-B1./(Tb+C1));
P20=exp(A2-B2./(Tb+C2));
y11=(gamma11*P10.*xf)./P;
y22=(gamma22*P20.*(1-xf))./P;

alfa=(y11/xf)/(y22/(1-xf)) 
hold off

%% McCabe
% Definerar initialvärden
xd=0.99;
xb=1-xd;
zf=0.5;

% Återflödesförhållande
Rmin=2.8569;
R=1.5*Rmin;

% Vi matar in mättad vätska i sep kol, alltså q = 1
q=1;

% Beräknar skärningen mellan feedlinjen och driftlinjen 
yi=(zf+xd*q/R)/(1+q/R);
xi=(-(q-1)*(1-R/(R+1))*xd-zf)/((q-1)*R/(R+1)-q);

% Skapar figur
figure(1);
hold on;
xlabel('Mol fraktion, vätskefas Eten');
ylabel('Mol fraktion, ångfas Eten');

% Plottar drift- och feedlinje samt jämviktskurva
plot(x1,y1,'r'), grid on; % Jämviktskurva
set(line([0  1],[0  1]),'Color',[0 1 0]); % Hjälplinje
set(line([xd xi],[xd yi]),'Color',[1 0 1]); % Övre driftlinje
set(line([zf xi],[zf yi]),'Color',[1 0 1]); % q-linje
set(line([xb xi],[xb yi]),'Color',[1 0 1]); % Nedre driftlinje

% Jämviktskurvan uträknad genom en "fitting" av uträknad data och matchad efter en 10:e grads polynom
p1 = 0.24912;p2 = -0.95581;p3 = 1.2782;
p4 = -0.13235;p5 = -1.8644;p6 = 3.2665;
p7 = -3.5287;p8 = 3.2073;p9 = -2.5254;
p10 = 2.0055;p11 = 7.1925e-08;
eqline = p1*x1.^10 + p2*x1.^9 + p3*x1.^8 + p4*x1.^7 + p5*x1.^6 + p6*x1.^5 + p7*x1.^4 + p8*x1.^3 + p9*x1.^2 + p10*x1 + p11;

%Förstärkardel
global y
bottnar=0;
i=1;
xp(1)=xd;
yp(1)=xd;
y=xd;

% Fungerar genom att rita ett streck horisontellt tills den möter jämviktskurvan, då börjar den rita lodrätt till den möter driftlinjen och förloppet startas om.
while xp(i)>xi
    xp(i+1)=fzero(@(x)equilib(x),0.5);
    yp(i+1)=R/(R+1)*xp(i+1)+xd/(R+1);
    y=yp(i+1);
    set(line([xp(i) xp(i+1)],[yp(i) yp(i)]),'Color',[0 0 1]);
    if (xp(i+1)>xi) set(line([xp(i+1) xp(i+1)],[yp(i) yp(i+1)]),'Color',[0 0 1]);
    end
        i=i+1;
        bottnar = bottnar + 1;
end

% Avdrivardel
SS=(yi-xb)/(xi-xb);
yp(i)=SS*(xp(i)-xb)+xb;
y=yp(i);
set(line([xp(i) xp(i)],[yp(i-1) yp(i)]),'Color',[0 0 1]);

% Fungerar på samma sätt som ovan.
while (xp(i)>xb)
    xp(i+1)=fzero(@(x)equilib(x),0.5);
    yp(i+1)=SS*(xp(i+1)-xb)+xb;
    y=yp(i+1);
    set(line([xp(i) xp(i+1)],[yp(i) yp(i)]),'Color',[0 0 1]);
    if (xp(i+1)>xb) set(line([xp(i+1) xp(i+1)],[yp(i) yp(i+1)]),'Color',[0 0 1]);
    end
    i=i+1;
    bottnar = bottnar + 1;
end        
hold off;
bottnar

%% Kondensor 
% Molflöden
FC2H41=0.99*1682;
FC2H61=0.01*1682;
Ftot1=1682;
Tm=283;

% Värmemängd
Hvap=14000;
q=Hvap*Ftot1;

% Värmekapacitet
CpC2H4in1= 5.409 + 0.1781*Tm - 6.938e-5*Tm^2 + 8.713e-9*Tm^3; %[J/mol*K]
CpC2H6in1= 3.806 + 0.1566*Tm - 8.324e-5*Tm^2 + 1.755e-8*Tm^3; %[J/mol*K]
CpC2H4ut1=67.24;
CpC2H6ut1=136.1;

% NTU
U=1000;
Cmin=FC2H61*CpC2H6in1+FC2H41*CpC2H6in1; %Cin
Cmax=FC2H61*CpC2H6ut1+FC2H41*CpC2H6ut1; %Cut
Cmin/Cmax;
NTU=2; % avläses 

% Area
A1=(NTU*Cmin)/U;

% Kostnad för drift
H= 8000; % antalet timmar på ett år
q_kwh=(q/10^3)*H; % kWH per år för vvx

kyl= 1*q_kwh; % Driftkostnad med kylmedium

% Förångare

% Molflöden
FC2H42=0.01*1945.56;
FC2H62=0.99*1945.56;
Ftot2=1945.56;
Tm=283;

% Värmekapacitet
CpC2H4ut2= 5.409 + 0.1781*Tm - 6.938e-5*Tm^2 + 8.713e-9*Tm^3; %[J/mol*K]
CpC2H6ut2= 3.806 + 0.1566*Tm - 8.324e-5*Tm^2 + 1.755e-8*Tm^3; %[J/mol*K]

CpC2H4in2=67.24;
CpC2H6in2=136.1;

% NTU
U=1000;
Cmax2=FC2H62*CpC2H6in2+FC2H42*CpC2H6in2 ;%Cin
Cmin2=FC2H62*CpC2H6ut2+FC2H42*CpC2H6ut2; %Cut
Cmin2/Cmax2;
NTU=2; % avläses 

% Area
A2=(NTU*Cmin2)/U;

% Kostnad

% Värmemängd
Hvap=15300;
q=Hvap*Ftot2
% Kostnad för drift
H= 8000; % antalet timmar på ett år
q_kwh=(q/10^3)*H; % kWH per år för vvx

kyl2= 0.05*q_kwh; % Driftkostnad med kylmedium;

% Kostnad för inköp
a=32000; b=70; % Kostnadsparametrar
S1=A1; % Storleksparameter
S2=A2 ;
n=1.2; % Utrustningsexponent

C1= a+b*S1^n *9.99; % Inköpskostnad år 2010 Kondensor
C2= a+b*S2^n *9.99; % Inköpskostnad år 2010 Förångare

disp(['Kondensor värmeväxlaryta    A = ' num2str(A1)   ' m2' ])
disp(['Driftkostnad Kondensor: = ' num2str(kyl/0.8)   ' kr/år' ])
disp(['Inköpskostnad för Kondensor = ' num2str(C1)   ' kr' ])
disp(['---------------------------------------------------------'])
disp(['Återkokare värmeväxlaryta    A = ' num2str(A2)   ' m2' ])
disp(['Driftkostnad Förångare: = ' num2str(kyl2/0.8)   ' kr/år' ])
disp(['Inköpskostnad för Förångare = ' num2str(C2)   ' kr' ])
disp(['---------------------------------------------------------'])

%% vvx 1
Tc=283;     %K
Th=906;     %K
U=1000;     %Kondensor/återkokare W/m^2K
% Värmekapaciteter
CPH2c=27.14 + 0.009274*Tc - 1.381*10^-5*Tc^2 + 7.645*10^-9*Tc^3;    %[J/molK]
CPH2Oc=32.24 + 0.001924*Tc + 1.055*10^-5*Tc^2 - 3.596*10^-9*Tc^3;   %[J/molK]
CPC2H6c= 3.806 + 0.1566*Tc - 8.324e-5*Tc^2 + 1.755e-8*Tc^3;      %[J/molK]
CPC2H4c= 5.409 + 0.1781*Tc - 6.938e-5*Tc^2 + 8.713e-9*Tc^3;    %[J/molK] 

CPH2h=27.14 + 0.009274*Th - 1.381*10^-5*Th^2 + 7.645*10^-9*Th^3;    %[J/molK]
CPH2Oh=32.24 + 0.001924*Th + 1.055*10^-5*Th^2 - 3.596*10^-9*Th^3;   %[J/molK]
CPC2H6h= 3.806 + 0.1566*Th - 8.324e-5*Th^2 + 1.755e-8*Th^3;      %[J/molK]
CPC2H4h= 5.409 + 0.1781*Th - 6.938e-5*Th^2 + 8.713e-9*Th^3;    %[J/molK]


Cmin= CPC2H6c*55.935 + CPC2H4c*56.5 + CPH2c*55.935 + CPH2Oc*1130; 
Cmax= CPC2H6h*55.935 + CPC2H4h*56.5 + CPH2h*55.935 + CPH2Oh*1130;
cmincmax=Cmin/Cmax;
NTU=4;
A=Cmin*NTU/U;

%kostnad Värmeväxlare
a = 3200;
b = 70;
n = 1.2;
S=A;
C = a + b.*S.^n;

disp(['VVX 1'])
disp(['Cmin= ' num2str(Cmin) ' [W/K]'])
disp(['Cmax= ' num2str(Cmax) ' [W/K]'])
disp(['Cmin/Cmax= ' num2str(cmincmax)])
disp(['NTU= ' num2str(NTU)])
disp(['Area= ' num2str(A) ' [m^2]'])
disp(['Inköpspris = ' num2str(C) ' [$]'])
disp(['---------------------------------------------------------'])

%% vvx 2
Tc=283;      %[K]
Th=375.9;    %[K]
U=1000;      %Kondensor/återkokare   1000 [W/m^2*K]
% Värmekapaciteter
CPH2c=27.14 + 0.009274*Tc - 1.381*10^-5*Tc^2 + 7.645*10^-9*Tc^3;    %[J/molK]
%CPH2Oc=32.24 + 0.001924*Tc + 1.055*10^-5*Tc^2 - 3.596*10^-9*Tc^3;   %[J/molK]
CPC2H6c= 3.806 + 0.1566*Tc - 8.324e-5*Tc^2 + 1.755e-8*Tc^3;      %[J/molK]
CPC2H4c= 5.409 + 0.1781*Tc - 6.938e-5*Tc^2 + 8.713e-9*Tc^3;    %[J/molK] 

CPH2h=27.14 + 0.009274*Th - 1.381*10^-5*Th^2 + 7.645*10^-9*Th^3;    %[J/molK]
%CPH2Oh=32.24 + 0.001924*Th + 1.055*10^-5*Th^2 - 3.596*10^-9*Th^3;   %[J/molK]
CPC2H6h= 3.806 + 0.1566*Th - 8.324e-5*Th^2 + 1.755e-8*Th^3;      %[J/molK]
CPC2H4h= 5.409 + 0.1781*Th - 6.938e-5*Th^2 + 8.713e-9*Th^3;    %[J/molK]

Cpc=CPC2H4c+CPH2c+CPC2H6c;
Cph=CPH2h+CPC2H6h+CPC2H4h;

mc=(55.935+56.5+55.935); %mol/s
mh=(55.935+56.5+55.935); %mol/s

Cmin=Cpc*mc; 
Cmax=Cph*mh;
cmincmax=Cmin/Cmax;
NTU=5;
A=Cmin*NTU/U;

%kostnad Värmeväxlare
a = 3200;
b = 70;
n = 1.2;
S=A;
C = a + b.*S.^n;

Q=mc*Cph*(Th-Tc)*10^-6;

disp(['VVX 2'])
disp(['Cmin= ' num2str(Cmin) ' [W/K]'])
disp(['Cmax= ' num2str(Cmax) ' [W/K]'])
disp(['Cmin/Cmax= ' num2str(cmincmax)])
disp(['NTU= ' num2str(NTU)])
disp(['Area= ' num2str(A) ' [m^2]'])
disp(['Inköpspris = ' num2str(C) ' [$]'])
disp(['---------------------------------------------------------'])

%% vvx 3
Tc=253;     %[K]
Th=293;     %[K]
U=200;      %Gas/vätska-värmeväxling   200 [W/m^2*K]
% Värmekapaciteter
CPH2c=27.14 + 0.009274*Tc - 1.381*10^-5*Tc^2 + 7.645*10^-9*Tc^3;    %[J/molK]
CPH2Oc=32.24 + 0.001924*Tc + 1.055*10^-5*Tc^2 - 3.596*10^-9*Tc^3;   %[J/molK]
CPC2H6c= 3.806 + 0.1566*Tc - 8.324e-5*Tc^2 + 1.755e-8*Tc^3;      %[J/molK]
CPC2H4c= 5.409 + 0.1781*Tc - 6.938e-5*Tc^2 + 8.713e-9*Tc^3;    %[J/molK] 

CPH2h=27.14 + 0.009274*Th - 1.381*10^-5*Th^2 + 7.645*10^-9*Th^3;    %[J/molK]
CPH2Oh=32.24 + 0.001924*Th + 1.055*10^-5*Th^2 - 3.596*10^-9*Th^3;   %[J/molK]
CPC2H6h= 3.806 + 0.1566*Th - 8.324e-5*Th^2 + 1.755e-8*Th^3;      %[J/molK]
CPC2H4h= 5.409 + 0.1781*Th - 6.938e-5*Th^2 + 8.713e-9*Th^3;    %[J/molK]

Cmin= CPC2H4c*55.935 + CPH2c*55.935 ; 
Cmax= CPC2H4h*55.935 + CPH2h*55.935 ;

cmincmax=Cmin/Cmax;
NTU=5.5;
A=Cmin*NTU/U;

%kostnad Värmeväxlare
a = 3200;
b = 70;
n = 1.2;
S=A;
C = a + b.*S.^n;

% Effekt från vvx 3
Tm=(293+253)/2;

CPH2m=27.14 + 0.009274*Tm - 1.381*10^-5*Tm^2 + 7.645*10^-9*Tm^3;    %[J/mol*K]
CPC2H4m= 160;    %[J/mol*K] nist

P_kolvate = (CPH2m*55.935 + CPC2H4m*56.5)*(293-253);

q = P_kolvate; %effekt ur vvx

drift=2.25*8000*q*10^-3;

disp(['VVX 3'])
disp(['Cmin= ' num2str(Cmin) ' [W/K]'])
disp(['Cmax= ' num2str(Cmax) ' [W/K]'])
disp(['Cmin/Cmax= ' num2str(cmincmax)])
disp(['NTU= ' num2str(NTU)])
disp(['Area= ' num2str(A) ' [m^2]'])
disp(['Inköpspris = ' num2str(C) ' [$]'])
disp(['Driftkostnad per år = ' num2str(drift) ' [kr]'])
disp(['---------------------------------------------------------'])

%% flash 1
k= 0.107-0.003*3;                   % [m/s]
rho_v= (1.3562+1.18)/2;             % [kg/m^3]
rho_l= 1000;                        % [kg/m^3]
V= (55.935*30e-3+56.5*28e-3)/rho_v;       % [m^3/s]
L=(1130*18e-3)/rho_l;               % [m^3/s]
tau=10*60;                          % [s]

u_t=k*sqrt((rho_l-rho_v)/rho_v);    % [m/s]

D=sqrt((4*V)/(pi*0.15*u_t));        % [m]
HL=(L*tau)/((pi/4)*D^2);            % [m]
H=HL+1.5*D;

P = 3300000; %(33 bar)
S = 134.3*1000*1000;    %N/mm^2 utifrån en temp på 283 kelvin eller ~50 farenheit
E = 1;
rho_ks = 7900;
a=11600 ; b=34; n=0.85;

t = (P*D)/((2*S*E)-(1.2*P));
D2 = D + 2*t;
V_skal = (H*pi/4)*(D2.^2-D.^2)+(2*(pi*(D2.^2)/4)*t);

m_skal = V_skal * rho_ks;

C=a+b*m_skal^n;

disp(['Flash 1'])
disp(['Höjd = ' num2str(H) '               [m]'])
disp(['Diameter = ' num2str(D) '           [m]'])
disp(['Kostnad inköp = ' num2str(C) ' [$]'])
disp(['---------------------------------------------------------'])

%% flash 2
k= 0.107-0.003*3;                   % [m/s]
rho_v= 0.08988;                     % [kg/m^3]
rho_l= 567;                         % [kg/m^3]
V=(56.5*2e-3)/rho_v;                % [m^3/s]     [(mol/s * kg/mol)/ kg/m^3]
L=(56.5*28e-3)/rho_l;               % [m^3/s]
tau=10*60;                          % [s]

u_t=k*sqrt((rho_l-rho_v)/rho_v);    % [m/s]

D=sqrt((4*V)/(pi*0.15*u_t));        % [m]
HL=(L*tau)/((pi/4)*D^2);            % [m]
H=HL+1.5*D;

P = 3300000; %(33 bar)
S = 134.3*1000*1000;    %[N/mm^2] utifrån en temp på 283 kelvin eller ~50 farenheit
E = 1;
rho_ks = 7900;
a=11600 ; b=34; n=0.85;

t = (P*D)/((2*S*E)-(1.2*P));
D2 = D + 2*t;
V_skal = (H*pi/4)*(D2.^2-D.^2)+(2*(pi*D2.^2/4)*t);

m_skal = V_skal * rho_ks;

C=a+b*m_skal^n;

disp(['Flash 2'])
disp(['Höjd = ' num2str(H) '               [m]'])
disp(['Diameter = ' num2str(D) '           [m]'])
disp(['Kostnad inköp = ' num2str(C) ' [$]'])
disp(['---------------------------------------------------------'])

%% Kompressor
% 1 = eten  2 = etan 3=H2
Patm=101325;                                                    %[Pa]
R=8.314;
F1  = 56.5;                                                     %[mol/s]
F2  = 55.935;                                                   %[mol/s]
F3  = 55.935;                                                   %[mol/s]
Cp1 = 50.4522;                                                  %[J/mol*K]
Cp2 = 41.8550;                                                  %[J/mol*K]
Cp3 = 28.8318;                                                  %[J/mol*K]
Ctot=F1*Cp1+F2*Cp2+Cp3*F3;                                      %[kJ/s*K]
cp=Ctot/(F1+F2+F3);
cv=cp-R;
kappa=cp/cv;                                                    %[Null]
Tin=283;                                                        %[K]
Put=3039750;                                                    %[Pa]
eta_is=0.8;

[Wtot,Qkyl,Akyltot,Tut]=kompressor(Ctot,kappa,Patm,Tin,Put,eta_is);
Qkyltot=2*Qkyl;                                                 %[W]

disp(['Wtot = ' num2str(Wtot) '        [W]'])
disp(['Qkyltot = ' num2str(Qkyltot) '     [W]'])
disp(['Akyltot = ' num2str(Akyltot) '         [m^2]'])
disp(['Inköps pris = ' num2str((((Akyltot/2).^1.2)*70+32000)*2+(((Wtot/3000).^0.6)*20000+580000)*3) ' [$]'])
disp(['Tut = ' num2str(Tut) '             [K]'])
disp(['---------------------------------------------------------'])
%Kompressorberäkningar för GKT-projekt
%Gäller för alkandehydrerings-projekten samt metanolprojektet.
%Tänk på att endast gaser tryckhöjs i kompressorer. För vätskor används
%pumpar.
%Skapad av: Elin Göransson, 2008-04-07
%
%Beräkningar bygger på antaganden om adiabatisk kompression och omräkning
%med isentropverkningsgrad för att få verkligt effektbehov.
%
%Kompressionen delas upp i tre steg med mellankylning pga den stora
%tryckökningen. Uppdelningen sker så att effektbehovet blir samma i varje
%steg, och kylningen emellan utformas så att man får samma temperatur in
%till varje kompressor.
%
%Funktionen:
%[Wtot,Qkyl,Akyltot,Tut]=kompressor(Ctot,kappa,Pin,Tin,Put,eta_is)
%
%Indata:
%Ctot[W/K]  = Summan av m*cp för alla komponenter i flödet, där m är
%             flödet i kg/s (alt mol/s) och cp är medelvärmekapaciviteten
%             över temperaturintervallet i kompressorn i J/(kgK) (alt J/(molK)).
%kappa []   = Kappatalet (viktat medelvärde av kappa för de olika
%             komponenterna)
%Pin [Pa]   = Ingående tryck till kompressorerna
%Tin [K]    = Ingående temperatur
%Put [Pa]   = Utgående tryck
%eta_is []  = Isentropverkningsgrad
%
%Utdata:
%Wtot [W]       = Totalt effektbehov för kompressionen.
%Qkyl [W]       = Kylbehov i mellankylare.
%Akyltot [m2]   = Total värmeväxlararea för mellankylare.
%Tut [K]        = Utgående temperatur.

%% Effekt från vvx 1 (från 906 K till 283 K)
Tm=(906+283)/2;
Tm1=(906+373)/2;
Tm2=(373+283)/2;

CPH2=27.14 + 0.009274*Tm - 1.381*10^-5*Tm^2 + 7.645*10^-9*Tm^3;    %[J/mol*K]
CPC2H6= 3.806 + 0.1566*Tm - 8.324e-5*Tm^2 + 1.755e-8*Tm^3;      %[J/mol*K]
CPC2H4= 5.409 + 0.1781*Tm - 6.938e-5*Tm^2 + 8.713e-9*Tm^3;    %[J/mol*K]

CPH2O1=32.24 + 0.001924*Tm1 + 1.055*10^-5*Tm1^2 - 3.596*10^-9*Tm1^3;   %[J/mol*K]
CPH2O2=32.24 + 0.001924*Tm2 + 1.055*10^-5*Tm2^2 - 3.596*10^-9*Tm2^3;   %[J/mol*K]
DH=40.66*10^3;

P_kolvate = (CPH2*55.935 + CPC2H6*55.935 + CPC2H4*56.5)*(906-283);
P_H2O = CPH2O1*1130*(906-373)+1130*DH+CPH2O2*1130*(373-283);

q = P_kolvate + P_H2O %effekt ur vvx
Driftskostnad = q/0.8/1000*8000*0.05 %per år och epsilon=0.8

%% Destillationsberäkning OBS!! Måste köras för att kunna räkna ut kostnad nedan
P = 3300000; %(33 bar)
D = 2.03;    %(m)
S = 88.9*1000*1000;    %N/mm^2 utifrån en temp på 283 kelvin eller ~50 farenheit
E = 1;
rho_ks = 7900;

t = (P*D)/((2*S*E)-(1.2*P));

d = 2.03;
D = 2.03 + 2*t;
h = 58 * 0.6096;

V_skal = (h*pi/4)*(D.^2-d.^2)+(2*(pi*D.^2/4)*t);
m_skal = V_skal * rho_ks;

%% Reaktortryckkärlberäkning
m_kat = 2500;    %kg
rho_kat = 1120; %kg/m^3
V_kat = m_kat/rho_kat;
P = 101325*1.1;
D = nthroot((2*V_kat)/pi,3);

%från linjär regression med hjälp av tabell från kurspm
S = 41.1*1000*1000;
E = 1;
rho_rf = 8000;

t_r = (P*D)/((2*S*E)-(1.2*P));

%för lågt använder minsta tabellvärde istället
tabrf = 0.07;
d = D
D = d + 2*tabrf;
h = 2*d

V_reak = (h*pi/4)*(D.^2-d.^2)+(2*(pi*D.^2/4)*t);
m_rskal = V_reak * rho_rf * 2; %antalet reaktorer

%% Kostnader
format long g

%Tryckkärl
ta = 11600;
tb = 34;
tn = 0.85;

%Klockbotten
kba = 340;
kbb = 640;
kbn = 1.9;
kbstrlk = 2.03;

%Värmeväxlare
vvxa = 3200;
vvxb = 70;
vvxn = 1.2;
vvx2strlk = 602.5;
vvx4strlk = 50.52;

a = [vvxa vvxa kba ta ta];
b = [vvxb vvxb kbb tb tb];
S = [vvx2strlk vvx4strlk kbstrlk m_skal m_rskal];
n = [vvxn vvxn kbn tn tn];

C = a + b.*S.^n

%%
function dFdW= odeeqq(W,F)
global R K1 P CP dHr0 dA dB dC dD Tref
FA=F(1);FB=F(2);FH2=F(3);FH2O=F(4);Ftot=F(1)+F(2)+F(3)+F(4); 
T=F(5);

%Partialtryck
PA=(F(1)/(Ftot))*P;                                                %[bar]
PB=(F(2)/(Ftot))*P;                                                %[bar]
PH2=(F(3)/(Ftot))*P;                                               %[bar]
PH20=(F(4)/(Ftot))*P;                                              %[bar]

%Värmekapacitet (eq 3.46 Termoboken)
CpA= 3.806 + 0.1566*F(5) - 8.324e-5*F(5)^2 + 1.755e-8*F(5)^3;      %[J/mol]
CpB= 5.409 + 0.1781*F(5) - 6.938e-5*F(5)^2 + 8.713e-9*F(5)^3;      %[J/mol]
CpH2=27.14 + 0.009274*F(5) - 1.381*10^-5*F(5)^2 + 7.645*10^-9*F(5)^3;    %[J/mol]
CpH2O=32.24 + 0.001924*F(5) + 1.055*10^-5*F(5)^2 - 3.596*10^-9*F(5)^3;   %[J/mol]
CP=[CpA CpB CpH2 CpH2O];                                           %[J/mol]

%Reaktionshastighet
k=(4.622*10^4*exp(-35.5*10^3/(R*F(5))));                %[mol/(kg cat.*s*bar)] 
Ke=1.48*10^7*exp(-144.8*10^3/(R*F(5)));                 %[bar]
ra=-(k*(PA-(PB*PH2)/Ke))/(1+K1*PB);

a=dA*(F(5)^1-Tref^1)/1;                                 
b=dB*(F(5)^2-Tref^2)/2;
c=dC*(F(5)^3-Tref^3)/3;
d=dD*(F(5)^4-Tref^4)/4;
dHr=dHr0 + a + b + c + d;

%ODE
dFAdW = ra; dFBdW = -ra; dFH2dW = -ra; dFH_2OdW=0;
dTdW =((-dHr)*(-ra))/(F(1)*CP(1)+F(2)*CP(2)+F(3)*CP(3)+FH2O*CP(4));

dFdW=[dFAdW    
      dFBdW    
      dFH2dW    
      dFH_2OdW
      dTdW];
end

function res = find_Tb(T,x1,gamma1,gamma2,A1,B1,C1,A2,B2,C2,P)
%Use T,x1,gamma1,gamma2,A1,B1,C1,A2,B2,C2,P to calculate y1 and y2
x2=1-x1;
P10=exp(A1-B1./(T+C1));
P20=exp(A2-B2./(T+C2));
y1=(gamma1*P10.*x1)./P;
y2=(gamma2*P20.*x2)./P;
res=y1+y2-1; 
end

function f=equilib(x)
global y
% Jämviktskurva estimerad från datapunkter.
p1 = 0.24912;p2 = -0.95581;p3 = 1.2782;
p4 = -0.13235;p5 = -1.8644;p6 = 3.2665;
p7 = -3.5287;p8 = 3.2073;p9 = -2.5254;
p10 = 2.0055;p11 = 7.1925e-08;
eqline = p1*x.^10 + p2*x.^9 + p3*x.^8 + p4*x.^7 + p5*x.^6 + p6*x.^5 + p7*x.^4 + p8*x.^3 + p9*x.^2 + p10*x + p11;
f=y-eqline;
end

function [Wtot,Qkyltot,Akyltot,Tut]=kompressor(Ctot,kappa,Pin,Tin,Put,eta_is)
%Tryckökning per steg.
P_step = (Put/Pin)^(1/3);  %[]
%Temperatur ut från varje kompressorsteg för isentrop kompression.
Tut_is = Tin*P_step^((kappa-1)/kappa);  %[K] 
%Verklig temperatur ut från varje kompressorsteg.
Tut = Tin + (Tut_is-Tin)/eta_is; %[K] 
%Erforderlig kompressoreffekt för ett kompressorsteg.
W = Ctot*(Tut-Tin); %[W] 
%Total erforderlig kompressoreffekt (3 steg).
Wtot = 3*W; %[W] 
%Erforderlig kyleffekt i 1 mellankylare
Qkyl = Ctot*(Tut-Tin);%[W] 
%Total erforderlig kyleffekt i mellankylare (2 st)
Qkyltot = 2*Qkyl; %[W] 
%Kylvattnets temperatur.
Tkv = -20+273.15; %[K] 
%Maximal temperatur som kylvattnet får värmas till
Tkvmax = 14+273.15; %[K] 
%Logaritmisk medeltemperaturdifferens.
deltaTlm = ((Tin-Tkv)-(Tut-Tkvmax))/log((Tin-Tkv)/(Tut-Tkvmax)); %[]
%U-värde för mellankylare (gas-vätska)
Ukyl = 200; %[W/(m2K)] 
%Värmeväxlararea för 1 mellankylare
Akyl = Qkyl/(Ukyl*deltaTlm); %[m2] 
%Total värmeväxlararea för mellankylarna.
Akyltot = 2*Akyl; %[m2] 
end

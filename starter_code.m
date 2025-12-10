[file,path] = uigetfile('*.h5','hdf5');
data = importAPDM(path,file,'SI-000646'); 
%%
freq = 128; 
acc = data(:,1:3);
gyr = data(:,4:6);
mag = data(:,7:9); 
quat = data(:,10:13); 
eul_kf = quatToEuler(quat);
eul_acc = quatToEuler(incAccel(acc)); 

%pitch
figure
plot([eul_acc(:,2),eul_kf(:,2)]); 
legend('acc','kf')
%roll
figure
plot([eul_acc(:,3),eul_kf(:,3)]); 
legend('acc','kf')

%%
q = ekf(gyr,acc,128,.005,.01); 
eul = quatToEuler(q);

plot([eul_kf(:,2),eul(:,2)]); 
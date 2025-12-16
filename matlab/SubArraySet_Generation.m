function [S, Nsub] = SubArraySet_Generation(B,N,M)
% Version: 1.0;
% Developed by Milan RC. delivered on 06/30/2017.

% B is assumed to be a Lsub x 2 matrix where Lsub is the size of the
% sub-array and 2 columns for (Iy,Iz) index positions

A=sum(B);
if A(1)==0 % vertical cluster
    step_M=size(B,1);
    step_N=1;
elseif A(2)==0 % horizontal cluster
    step_N=size(B,1);
    step_M=1;
end

S=[];

for kk=min(M):step_M:max(M)
    for hh=min(N):step_N:max(N)
        Bshift=[B(:,1)+hh,B(:,2)+kk];
        check=isempty(find(Bshift(:,1)>max(N) | Bshift(:,1)<min(N) | Bshift(:,2)>max(M) | Bshift(:,2)<min(M)));
        if (check)
            S=[S,Bshift];
        end
    end
end

Nsub=size(S,2)/2;

end

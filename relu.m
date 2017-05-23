m=importdata('train.data',',');
size1=size(m)
ss=size1(1)
xdata=zeros(ss,127);
result=zeros(ss,1);
for i=[1:size1]
    xdata(i,1)=1;
    y=strsplit(m{i},',');
    for j=[1:126]
        
        xdata(i,j+1)=y{j}-48;
    end
    if (strcmp('win',y{127})==1)
        result(i)=1;
    elseif(strcmp('loss',y{127})==1)
        result(i)=2;
    else
        result(i)=3;
    end
        
end

%theta=zeros(127,100);
minm=0.01;
maxm=0.02;
theta=minm+rand(127,100)*(maxm-minm);


phi=minm+rand(100,3)*(maxm-minm);
hiddeninp=zeros(100,1);
output=zeros(3,1);
iter=0;
n=0.005
while(iter<200)
    iter=iter+1
    
    for index=[1:ss]
        x=xdata(index,:);
        hidden=theta'*x';
        
        
        vals=log(1.0+exp(hidden));
        otpt=phi'*vals;
        finalotpt=log(1.0+exp(otpt));
        if(result(index)==1)
            res=[1;0;0];
        elseif(result(index)==2)
            res=[0;1;0];
        else
            res=[0;0;1];
        end
        tmp=-1*(res-finalotpt).*(1-exp(-finalotpt));
        deltaphi=vals*tmp';
        size(deltaphi);
        
        tmp2=zeros(100,1);
        for k=[1:3]
           tmp2=tmp2+tmp(k)*(phi(:,k).*(1-exp(-vals))) ;
        end
        deltatheta=x'*tmp2';
        phi=phi-n*deltaphi;
        theta=theta-n*deltatheta;
        
        
       
    end
end

count=0;
for id=[1:ss]
        xnew=xdata(id,:);
        hiddennew=theta'*xnew';
        
        valsnew=log(1.0+exp(hiddennew));
        otptnew=phi'*valsnew;
        finalotptnew=log(1.0+exp(otptnew));
       idx= find(finalotptnew == max(finalotptnew(:)));
       if(idx==result(id))
           count=count+1;
       end
        
        
       
end
      
 acc=(count*100.0)/ss
 
 
 
 
 
 
m2=importdata('test.data',',');
size2=size(m2);
ss2=size2(1);
xdata2=zeros(ss2,127);
result2=zeros(ss2,1);
for i=[1:size2]
    xdata2(i,1)=1;
    y2=strsplit(m2{i},',');
    for j=[1:126]
        
        xdata2(i,j+1)=y2{j}-48;
    end
    if (strcmp('win',y2{127})==1)
        result2(i)=1;
    elseif(strcmp('loss',y2{127})==1)
        result2(i)=2;
    else
        result2(i)=3;
    end
        
end
 
 
count2=0;
for id2=[1:ss2]
        xnew2=xdata2(id2,:);
        hiddennew2=theta'*xnew2';
        
        valsnew2=log(1.0+exp(hiddennew2));
        otptnew2=phi'*valsnew2;
        finalotptnew2=log(1.0+exp(otptnew2));
       idx2= find(finalotptnew2 == max(finalotptnew2(:)));
       if(idx2==result2(id2))
           count2=count2+1;
       end
        
        
       
end


testaccuracy=(count2*100.0)/ss2
      
 

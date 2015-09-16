function [distance, insert, delete, substitute, correct] = editDis(t,r)
m = numel(t); % groundtruth
n = numel(r); % recognition

insert = 0;          % 0 ��ʾ����
delete = 0;          % 1 ��ʾɾ��
substitute = 0;      % 2 ��ʾ���
correct = 0;         % 3 ��ʾ��ȷ

% �ۻ��������
dis = toeplitz(1:m+1, 1:n+1) - 1; %��ʼ��
route = ones(m+1,n+1,3);          %·������

for i=2:m+1
    for j=2:n+1
        tij = ne(t(i-1), r(j-1)); % 1:����ȣ� 0�����
        dis(i,j) = min(min(dis(i-1,j)+1, dis(i,j-1)+1), dis(i-1,j-1)+tij);
        if dis(i,j) == dis(i-1,j)+1 
            route(i,j,1) = i-1; % ǰһ��·�����i����
            route(i,j,2) = j;   % ǰһ��·�����j����
        elseif dis(i,j) == dis(i,j-1)+1
            route(i,j,1) = i;
            route(i,j,2) = j-1;
        elseif dis(i,j) == dis(i-1,j-1)+tij
            route(i,j,1) = i-1;
            route(i,j,2) = j-1;
        end
    end 
end

% ����
i=m+1;
j=n+1;
while i>=1 && j>=1
    i_back = route(i,j,1);
    j_back = route(i,j,2);
    if j_back == j && i_back~=i  %�������·����ֱ����ʾi-i_back��delete����
        delete = delete + abs(i-i_back);
    end
    if i_back==i && j_back~=j   %�������·��ˮƽ����ʾj-j_back��insert����
        insert = insert + abs(j-j_back);
    end
    if i_back == i-1 && j_back == j-1 %�������·��б����Ҫ�������������ж�Ӧ��ֵ�Ƿ���ȣ���ȷ����corr����sub
        if t(i_back)==r(j_back)
            correct = correct +1;
        else
            substitute = substitute +1;
        end
    end
    i = i_back;
    j = j_back;
    if i==1 && j==1
        break;
    end
end

distance = dis(m+1,n+1);

end

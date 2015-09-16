function [distance, insert, delete, substitute, correct] = editDis(t,r)
m = numel(t); % groundtruth
n = numel(r); % recognition

insert = 0;          % 0 表示插入
delete = 0;          % 1 表示删除
substitute = 0;      % 2 表示替代
correct = 0;         % 3 表示正确

% 累积距离矩阵
dis = toeplitz(1:m+1, 1:n+1) - 1; %初始化
route = ones(m+1,n+1,3);          %路径矩阵

for i=2:m+1
    for j=2:n+1
        tij = ne(t(i-1), r(j-1)); % 1:不相等， 0：相等
        dis(i,j) = min(min(dis(i-1,j)+1, dis(i,j-1)+1), dis(i-1,j-1)+tij);
        if dis(i,j) == dis(i-1,j)+1 
            route(i,j,1) = i-1; % 前一个路径点的i坐标
            route(i,j,2) = j;   % 前一个路径点的j坐标
        elseif dis(i,j) == dis(i,j-1)+1
            route(i,j,1) = i;
            route(i,j,2) = j-1;
        elseif dis(i,j) == dis(i-1,j-1)+tij
            route(i,j,1) = i-1;
            route(i,j,2) = j-1;
        end
    end 
end

% 回溯
i=m+1;
j=n+1;
while i>=1 && j>=1
    i_back = route(i,j,1);
    j_back = route(i,j,2);
    if j_back == j && i_back~=i  %如果回溯路径竖直，表示i-i_back个delete出现
        delete = delete + abs(i-i_back);
    end
    if i_back==i && j_back~=j   %如果回溯路径水平，表示j-j_back个insert出现
        insert = insert + abs(j-j_back);
    end
    if i_back == i-1 && j_back == j-1 %如果回溯路径斜向，需要考察两个数组中对应的值是否相等，来确定是corr或者sub
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

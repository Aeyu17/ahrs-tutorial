function q = quatMultiply(q1, q2)
% From Kok et al. A.1 - Quaternion Algebra
% Alternatively, eq. 3.27
q = [q1(1)*q2(1) - q1(2:4)'*q2(2:4);...
     q1(1)*q2(2:4) + q2(1)*q1(2:4) + cross(q1(2:4),q2(2:4))];

end
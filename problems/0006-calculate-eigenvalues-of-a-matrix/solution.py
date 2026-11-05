import math
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    
    trace = a + d
    det = a * d - b * c  # 修正：行列式公式应为 a*d - b*c
    
    # 修正：补充判别式 (discriminant) 的计算
    discriminant = trace**2 - 4 * det
    
    if discriminant < 0:
        # 如果你需要处理复数特征值，可以在这里抛出异常或返回 complex
        pass 
        
    sqrt_disc = math.sqrt(discriminant)
    
    eigenvalues1 = 0.5 * (trace - sqrt_disc)
    eigenvalues2 = 0.5 * (trace + sqrt_disc)

    return [eigenvalues2, eigenvalues1]
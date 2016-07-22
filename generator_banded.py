import sys
import numpy as np
import matplotlib.pylab as plt
import scipy.io as IO
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import scipy.linalg as la
from array import array

def create_tridiagonal(n):
    '''
    Crea un sistema tridiagonal de dimension N
    '''
    np.random.seed(314) # try to make it reproducible


    A = sparse.diags(np.random.rand(n) +1.0, 0) + \
        sparse.diags(np.random.rand(n-1), 1) + \
        sparse.diags(np.random.rand(n-1),-1)

    A = A.tocsr()

    # create rhs
    # B  = np.ones(shape=(n, nrhs), dtype=np.float64)

    return (A)

def create_pentadiagonal(n):
    '''
    Crea un sistema tridiagonal de dimension N
    '''
    np.random.seed(314) # try to make it reproducible


    A = sparse.diags(np.random.rand(n) +1.0, 0) + \
        sparse.diags(np.random.rand(n-1), 1) + \
        sparse.diags(np.random.rand(n-1),-1) + \
        sparse.diags(np.random.rand(n-2), 2) + \
        sparse.diags(np.random.rand(n-2),-2)

    A = A.tocsr()

    # create rhs
    # B  = np.ones(shape=(n, nrhs), dtype=np.float64)

    return (A)



def create_banded(n):
    '''
    Crea un sistema tridiagonal de dimension N
    '''
    np.random.seed(314) # try to make it reproducible


    A = sparse.diags(np.random.rand(n) +1.0, 0) + \
        sparse.diags(np.random.rand(n-2), 2) + \
        sparse.diags(np.random.rand(n-3),-3) + \
        sparse.diags(np.random.rand(n-4),-4)

    A = A.tocsr()

    # create rhs
    # B  = np.ones(shape=(n, nrhs), dtype=np.float64)

    return (A)


def export_csr2bin ( M, outputfilename):
    '''
    Exporta el sistema Mx=b con multiples lados derechos al formato definido para el SPIKE

    @M: matrix sparse de entrada
    @B: lado derecho del sistema
    '''
    output_file = open( outputfilename, 'wb')

    if sparse.isspmatrix_csr( M ) == False:
        print 'Warning, input matrix is not CSR!'
        M = M.tocsr()


    # eliminamos posibles ceros fruto de la extraccion o la incorrecta creacion de la matriz
    # M.eliminate_zeros()

    # flags para los tipos de datos empleados
    DOUBLE_t   = 0
    COMPLEX_t  = 1


    if( type(M.data[0]) == np.complex128 ):
        print 'Datatype is complex'
        datatype = COMPLEX_t
    else:
        print 'Datatype is double'
        datatype = DOUBLE_t


    # print 'rhs dimensions', len(B.shape)

    # if (len(B.shape) == 1):
    # 	nrhs = 1
    # else:
    # 	nrhs = B.shape[1]

    print 'System features'
    print 'Coefficients datatype   : ', type( M.data[0] )
    print 'System dimension        : ', M.shape
    print 'nnz elements            : ', M.nnz
    # print 'Numero de RHS           : ', nrhs

    # guardamos el numero de filas de la matriz de coeficientes
    float_array = array('i', [ M.shape[0] ])
    float_array.tofile(output_file)

    # guardamos el numero de columnas de la matrix de coeficientes
    #float_array = array('i', [ M.shape[1] ])
    #float_array.tofile(output_file)

    # guardamos el numero de elementos no nulos
    float_array = array('i', [M.nnz])
    float_array.tofile(output_file)

    # guardamos el numero de lados derechos del sistema
    #float_array = array('i', [nrhs] )
    #float_array.tofile(output_file)

    # guardamos el tipo de dato de los coeficientes y el rhs (0 = double, 1 = double complex)
    float_array = array('i', [datatype])
    float_array.tofile(output_file)


    if datatype == COMPLEX_t:
        # guardamos la parte real de la matriz de coeficientes
        float_array = array('d', M.data.real )
        float_array.tofile(output_file)

        # guardamos la parte imaginaria de la matriz de coeficientes
        float_array = array('d', M.data.imag )
        float_array.tofile(output_file)

    else:
        # guardamos los coeficientes del sistema como doubles
        float_array = array('d', M.data )
        float_array.tofile(output_file)


    # guardamos los inidices de columnas
    float_array = array('i', M.indices )
    float_array.tofile(output_file)

    # guardamos los indices de punteros a filas
    float_array = array('i', M.indptr )
    float_array.tofile(output_file)

    # guardamos el vector B
    # if datatype == COMPLEX_t:
    #     float_array = array('d',  np.ravel( B.real, order='F') )
    #     float_array.tofile(output_file)

    #     float_array = array('d',  np.ravel( B.imag, order='F') )
    #     float_array.tofile(output_file)

    # else:
    #     float_array = array('d',  np.ravel( B, order='F') )
    #     float_array.tofile(output_file)

    # cerramos el fichero
    output_file.close()

    print 'Linear system was sucessfully exported: ', outputfilename

if __name__ == "__main__":

    np.set_printoptions(precision=3, linewidth=200)

    if len(sys.argv) < 2 or float(sys.argv[1]) <= 0:
        raise ValueError('Dimension of the system must be supplied, and must be positive!')

    dim = 20000 #int(sys.argv[1])

    A = create_pentadiagonal( dim )
    export_csr2bin( A, "Tests/spike/small.bin")

    #Block = A[0:5,0:5]
#
#    #b        = A[0:5,5:7].todense()
#
#    #print 'RHS of the linear system is:'
#    #print b
#
#    #LU       = sla.splu( Block.tocsc() );
#    #solution = LU.solve(b)
#    #print solution
#
#    #print 'Norm of b:        %E' %(la.norm(b))
#    #print 'Absolute residual %E' %(la.norm( Block*solution - b))
    #print 'Relative residual %E' %(la.norm( Block*solution - b) / la.norm(b))



    print 'End of the program'

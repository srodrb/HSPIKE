import numpy as np
from scipy.linalg import norm
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from array import array
from timeit import default_timer as timer


def spike_naive(A, f, ku, kl, partitions):
    '''
    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''
    import scipy.sparse.linalg as sla
    from numpy import zeros, empty
    from scipy.sparse import eye, isspmatrix_csc

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    print 'A dim %d number of rhs %d' % (mdim, nrhs)

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    print 'Block dimension %d, subblockdim %d' % (blockdim, subblockdim)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y = zeros(shape=(mdim, nrhs), dtype=dprec)
    x = zeros(shape=(mdim, nrhs), dtype=dprec)

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        print 'A[%d:%d] sub[%d,%d]' % (bs, be, rs, re)

        if p is 0:
            print 'Factorizing first block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para B->V
            V = lu.solve(A[bs:be, be:be+ku].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            print 'Factorizing last block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para W->C
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            print 'Factorizing middle block (p=%d)' % (p)

            print 'Block dimensions', be - bs

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            V = lu.solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = lu.solve(f[bs:be, :])
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    print 'Reduced system'
    print R.todense()

    print 'RHS of the reduced system'
    print yr


    lu = sla.splu(R.tocsc())
    xred = lu.solve(yr)

    print 'Solution of the reduced system'
    print xred

    print 'Residual for this solution %E' %( norm( R.dot(xred) - yr ) / norm(yr) )

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :])
        elif p is (partitions - 1):
            x[bs+ku:be-kl, :] = (y[bs+ku:be-kl, :] - A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))
        else:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - (A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :]) + A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))

    return (x, total_nnz)

def spike_implicit_vw(A, f, ku, kl, partitions, fully_implicit=True):
    '''
    Esta version del SPIKE emplea un uso implicito de las matrices Vi, Wi.

    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''

    import scipy.sparse.linalg as sla
    from numpy import zeros, empty, eye, empty_like
    from scipy.sparse import eye, isspmatrix_csc

    solver_start_t = timer()

    permc_spec = 'COLAMD'

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R  = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y  = zeros(shape=(mdim, nrhs), dtype=dprec)
    x  = zeros(shape=(mdim, nrhs), dtype=dprec)
    Ai = []

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para B->V
            V = Ai[p].solve(A[bs:be, be:be+ku].todense())

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para W->C
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            V = Ai[p].solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    R = R.tocsc()

    start_t = timer()
    lu = sla.splu(R)
    xred = lu.solve(yr)
    end_t = timer()
    print 'Factorization and solve time for the reduced system %.5f' % (end_t - start_t)
    

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    if fully_implicit is True:
        fl = empty_like(f[0:blockdim, :])

        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                fl = f[bs:be, :].copy()
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            else:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
    else:
        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]))[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]
            else:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]) - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]

    solver_end_t = timer()

    print 'SPIKE inner timer: %.5f seconds' % (solver_end_t - solver_start_t)


    return (x, total_nnz)

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
    #   nrhs = 1
    # else:
    #   nrhs = B.shape[1]

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


if __name__ == '__main__':
    print 'SPIKE solver'
    np.set_printoptions(precision=5, threshold=300, linewidth=300)

    # generamos la semilla del generador de numeros aleatorios
    np.random.seed(314)

    # numero de particiones empleadas en el primer nivel
    p = 3
    N = 15

    #A = create_banded_matrix(N, [0, 90, 50, -40, -90])
    A = create_pentadiagonal(N)
    export_csr2bin( A, "../Tests/spike/penta_15.bin")

    nrhs = 1

    print 'dim(A) %d, (ku,kl) = (%d,%d) processes %d' % (A.shape[0], 2, 2, p)

    # b = np.random.rand(N*nrhs).reshape(N, nrhs)
    b = np.ones(N).reshape(N, nrhs)
    b_superlu = b.copy()
    b_spike = b.copy()


    # analysis.performance_analysis(A, b[:,0:2], 'analysis.rhs.1', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:45], 'analysis.rhs.45', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:90], 'analysis.rhs.90', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:180], 'analysis.rhs.180', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:360], 'analysis.rhs.360', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:720], 'analysis.rhs.720', pmin=1, pmax=50)
    # exit()

    print 'Using semi-implict SPIKE solver now...'
    x_spike, spike_nnz = spike_implicit_vw (A.copy(), b_spike, 2, 2, p, fully_implicit=True )


    for i in range( min(nrhs,10)):
        print '%.2d-th RHS - SPIKE error: %.2E' %(i, norm(b[:, i] - A.dot(x_spike[:, i])))

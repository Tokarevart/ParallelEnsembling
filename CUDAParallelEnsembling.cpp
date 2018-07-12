// Размерность пр-ва
#define DIM 3
// Кол-во узлов элемента.
#define Npt 4
// Кол-во потоков в блоке
#define BLOCK_SIZE 256


// glM[i][j] - глобальная матрица жёсткости.
//     i     - строка, 
//        j  - столбец.
// Размер: DIM*nodesNum x DIM*nodesNum
//
// Np[i][j] - глобальный номер узла.
//    i     - глобальный номер элемента,
//       j  - локальный  номер узла.
// Размер: elemsNum x Npt
//
// K[i][j][k] - значение из локальной матрицы.
//   i        - глобальный номер элемента,
//      j     - строка,
//         k  - столбец.
// Размер: elemsNum x 2*DIM*Npt x 2*DIM*Npt
//
// elemsNum - общее кол-во элементов.
//
// nodesNum - общее кол-во узлов.
//
// Nconc[i] - кол-во элементов, содержащих i-ый узел.
//       i  - глобальный номер узла.
// Размер: nodesNum
//
// maxNconc - наибольшее кол-во элементов, содержащих какой-либо узел.
// maxNconc = max(Nconc, nodesNum);
// где max - функция, вычисляющая наибольшее значение массива.
//
// Elem_Conc[i][j] - глобальный номер элемента.
//           i     - глобальный номер узла,
//              j  - локальный номер элемента.
// Размер: nodesNum x maxNconc
//
// No_Corresp[i][j] - локальный номер узла.
//            i     - глобальный номер узла,
//               j  - локальный номер элемента.
// Размер: nodesNum x maxNconc

// Считается, что входные матрицы представляют собой каждая единый неразрывный блок памяти.


template<class T>
__global__
void ParallelEnsemble(T* d_glM, size_t* d_Np, T* d_K, size_t* d_Nconc, size_t* d_Elem_Conc, size_t* d_No_Corresp, size_t* d_nodesNum, size_t* d_maxNconc)
{
	size_t nodesNum = *d_nodesNum;
	size_t maxNconc = *d_maxNconc;

	size_t node_loc_num;
	size_t elem_glob_num;
	size_t ii, jj;
	T*  block[DIM];
	__shared__ T sharedBlockArr[BLOCK_SIZE][DIM][DIM];
	T* sharedBlock = (T*)sharedBlockArr[threadIdx.x];

	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < nodesNum)
	{
		size_t twoDIMNpt = 2 * DIM * Npt;
		for (size_t j = 0; j < d_Nconc[index]; j++)
		{
			node_loc_num  = d_No_Corresp[index * maxNconc + j];
			elem_glob_num = d_Elem_Conc [index * maxNconc + j];

			ii = DIM * d_Np[elem_glob_num * Npt + node_loc_num];
			for (int k = 0; k < Npt; k++)
			{
				block[0] = &d_K[elem_glob_num * twoDIMNpt * twoDIMNpt +  DIM * node_loc_num      * twoDIMNpt + DIM * k];
				block[1] = &d_K[elem_glob_num * twoDIMNpt * twoDIMNpt + (DIM * node_loc_num + 1) * twoDIMNpt + DIM * k];
				block[2] = &d_K[elem_glob_num * twoDIMNpt * twoDIMNpt + (DIM * node_loc_num + 2) * twoDIMNpt + DIM * k];

				sharedBlock[      0] = block[0][0]; sharedBlock[          1] = block[0][1]; sharedBlock[          2] = block[0][2];
				sharedBlock[    DIM] = block[1][0]; sharedBlock[    DIM + 1] = block[1][1]; sharedBlock[    DIM + 2] = block[1][2];
				sharedBlock[2 * DIM] = block[2][0]; sharedBlock[2 * DIM + 1] = block[2][1]; sharedBlock[2 * DIM + 2] = block[2][2];

				// Проверка не заполнен ли блок нулями.
				if (!IsZero3x3<T>(sharedBlock))
				{
					jj = DIM * d_Np[elem_glob_num * Npt + k];

					Add3x3<T>(d_glM, nodesNum, ii, jj, sharedBlock);
				}
			}
		}
	}
}

template<class T>
__device__
bool IsZero3x3(T* block)
{
	for (int i = 0; i < 3; i++)
	{
		if ((block[i * DIM] < -1e-5 || block[i * DIM] > 1e-5))
		{
			return false;
		}
		if ((block[i * DIM + 1] < -1e-5 || block[i * DIM + 1] > 1e-5))
		{
			return false;
		}
		if ((block[i * DIM + 2] < -1e-5 || block[i * DIM + 2] > 1e-5))
		{
			return false;
		}
	}

	return true;
}

template<class T>
__device__
void Add3x3(T* &d_glM, size_t &nodesNum, size_t &ii, size_t &jj, T* block)
{
	size_t index = ii * DIM * nodesNum + jj;
	d_glM[index] += block[      0]; d_glM[index + 1] += block[          1]; d_glM[index + 2] += block[          2];

	index += DIM * nodesNum;
	d_glM[index] += block[    DIM]; d_glM[index + 1] += block[    DIM + 1]; d_glM[index + 2] += block[    DIM + 2];

	index += DIM * nodesNum;
	d_glM[index] += block[2 * DIM]; d_glM[index + 1] += block[2 * DIM + 1]; d_glM[index + 2] += block[2 * DIM + 2];
}
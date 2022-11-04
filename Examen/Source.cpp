#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <math.h>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

//Definicion de funciones
Mat pasarKernel(Mat imagenAmpleada, Mat kernel, int filasImagen, int columnasImagen, int tamKernel);
Mat expandirImg(Mat imagen, int tamKernel);
Mat baseGx();
Mat baseGy();
Mat supresion(Mat sobel, Mat imagenFiltroGauss, Mat matrizGx, Mat matrizGy);
Mat umbralizado(Mat sup, float umbralBajo, float umbralAlto);

int main()
{
	char NombreImagen[] = "Lenna.png";

	//Matrices para img original y a escala de grises 
	Mat imagen;
	Mat imagenGrises;

	int tamExpancion = 0;
	int tamKernel = 0;
	int centro = 0;
	int x, y;
	int i, j;

	float sigma = 0;
	float valorForm = 0;

	// Constantes
	float pi = 3.14159265358979323846;
	float e = 2.71828182845904523536;

	double azul, verde, rojo;

	// Pedimos el valor del kernel y sigma
	cout << "Ingresa un tamaño impar del kernel" << endl;
	cin >> tamKernel;

	cout << "Ingresa el valor de sigma" << endl;
	cin >> sigma;

	// Matriz para el Kernel
	Mat kernel(tamKernel, tamKernel, CV_32F); 
	
	//Buscamos el centro del kernel
	centro = (tamKernel - 1) / 2; 
	cout << "Centro: " << centro << "\n" << endl;

	//Kernel con coordenadas 
	cout << "\nCoordenadas\n" << endl;
	for (i = 0; i < tamKernel; i++){
		y = -1 * (i - centro);

		for (j = 0; j < tamKernel; j++){
			x = j - centro;

			valorForm = (1 / (2 * pi * sigma * sigma)) * pow(e, -((x * x + y * y) / (2 * sigma * sigma)));
			kernel.at<float>(Point(j, i)) = valorForm;

			cout << "\t(" << x << "," << y << ")";
		}
		cout << endl;
	}
	cout << "\nKernel Gauss\n" << endl;
	cout << kernel << "\n" << endl;

	// Lectura de la imagen
	imagen = imread(NombreImagen); 
	if (!imagen.data)
	{
		cout << "Error al cargar la imagen: " << NombreImagen << endl;
		exit(1);
	}

	//Filas y columnas de la imagen original
	int filasOrig = imagen.rows; 
	int columnasOrig = imagen.cols; 

	
	//Obtenemos la imagen original en escala de grises para poder trabajarla
	Mat imgGrises(filasOrig, columnasOrig, CV_8UC1);
	for (i = 0; i < filasOrig; i++)
	{
		for (j = 0; j < columnasOrig; j++)
		{
			azul = imagen.at<Vec3b>(Point(j, i)).val[0];  // Canal de color azul
			verde = imagen.at<Vec3b>(Point(j, i)).val[1]; // Canal de color verde
			rojo = imagen.at<Vec3b>(Point(j, i)).val[2];  // Canal de color rojo

			imgGrises.at<uchar>(Point(j, i)) = uchar(0.299 * rojo + 0.587 * verde + 0.114 * azul);
		}
	}
	// cout << imagenGrises << "\n" << endl;

	/*******************Inicia procesos*****************/
	// Expandimos imagen
	Mat imagenAmpleada = expandirImg(imgGrises, tamKernel);

	//Sumamos todos los valores del kernel para aplicar normalizacion 
	float sumaKernel = 0;
	for (int i = 0; i < tamKernel; i++){
		for (int j = 0; j < tamKernel; j++){
			sumaKernel += kernel.at<float>(Point(j, i));
		}
	}
	cout << "Suma kernel: " << sumaKernel << "\n" << endl;

	Mat imagenFiltroGauss = pasarKernel(imagenAmpleada, kernel, filasOrig, columnasOrig, tamKernel) / sumaKernel;
	cout << "Tamaño imagen filtro Gauss: " << imagenFiltroGauss.rows << "x" << imagenFiltroGauss.cols << endl;

	//Equaliza la img con filtro Gaussiano
	Mat equalizada;
	cv::equalizeHist(imagenFiltroGauss, equalizada);


	/***********Inicia proceso para filtro sobel************/

	// Creamos los kernel Gx y Gy
	Mat kerGx = baseGx(); 
	Mat kerGy = baseGy(); 
	cout << "\nKenrnelGx\n" << kerGx << "\n" << endl;
	cout << "\nKenrnelGy\n" << kerGy << "\n" << endl;

	// Expandimos imagen nuevamente
	Mat imgAmp2 = expandirImg(imagenFiltroGauss, 3); //equalizada

	// Pasamos los dos kernel Gx y Gy ppor la imagen ampleada del filtro de gauss
	//Obtenemos la matrizGx y la matrizGy
	Mat matrizGx = pasarKernel(imgAmp2, kerGx, imgAmp2.rows, imgAmp2.cols, 3);
	Mat matrizGy = pasarKernel(imgAmp2, kerGy, imgAmp2.rows, imgAmp2.cols, 3);

	//Obtenemos la matriz |G|
	Mat G(imagenFiltroGauss.rows, imagenFiltroGauss.cols, CV_8UC1);
	for (int i = 0; i < imagenFiltroGauss.rows; i++)
	{
		for (int j = 0; j < imagenFiltroGauss.cols; j++)
		{
			//Formula para sacar |G|
			G.at<uchar>(Point(j, i)) = sqrt(pow(matrizGx.at<uchar>(Point(j, i)), 2) + pow(matrizGy.at<uchar>(Point(j, i)), 2));
		}
	}
	cout << "Tamaño imagen |G|: " << G.rows << "x" << G.cols << endl;

	//Expandimos la de sobel 1 pixel 
	Mat sobel = expandirImg(G, 1);

	//Obtenemos la matriz de supresion
	Mat sup = supresion(sobel, imagenFiltroGauss, matrizGx, matrizGy);

	/***********Inicia proceso de umbralizado************/
	float umbralBajo = 0.25; //0.25, 0.45
	float umbralAlto = 0.45;

	//Obtenemos la mariz ubralizada
	Mat umbral = umbralizado(sup, umbralBajo, umbralAlto);
	cout << "Tamaño de imagen Canny: " << umbral.rows << "x" << umbral.cols << endl;

	/***********Creamos las ventanas donde se visualizan ls imagenes***********/
	namedWindow("Imagen original", WINDOW_AUTOSIZE); 
	imshow("Imagen original", imagen);

	namedWindow("Imagen escala de grises", WINDOW_AUTOSIZE);
	imshow("Imagen escala de grises", imgGrises);

	//namedWindow("ImagenaAmpleada", WINDOW_AUTOSIZE);
	//imshow("ImagenAmpleada", imagenAmpleada);

	/*namedWindow("ImagenaAmpleada2", WINDOW_AUTOSIZE);
	imshow("ImagenAmpleada2", imagenAmpleada2);*/

	namedWindow("Imagen suavizada", WINDOW_AUTOSIZE);
	imshow("Imagen suavizada", imagenFiltroGauss);

	namedWindow("Imagen equalizada", WINDOW_AUTOSIZE);
	imshow("Imagen equalizada", equalizada);

	//namedWindow("ImagenGX", WINDOW_AUTOSIZE);
	//imshow("ImagenGX", matrizGx);

	//namedWindow("ImagenGY", WINDOW_AUTOSIZE);
	//imshow("ImagenGY", matrizGy);

	namedWindow("Imagen filtro sobel (G)", WINDOW_AUTOSIZE);
	imshow("Imagen filtro sobel (G)", G);

	//namedWindow("Imagen supresion", WINDOW_AUTOSIZE);
	//imshow("Imagen supresion", sup);


	namedWindow("Imagen deteccion borde Canny", WINDOW_AUTOSIZE);
	imshow("Imagen deteccion borde Canny", umbral);

	waitKey(0);

	return 1;
}

Mat pasarKernel(Mat imagenAmpleada, Mat kernel, int filasImagen, int columnasImagen, int tamKernel){

	Mat conFiltro(filasImagen, columnasImagen, CV_8UC1);
	float suma = 0;
	int indiceAmpi = 0;
	int indiceAmpj = 0;

	for (int i = 0; i < filasImagen; i++){
		for (int j = 0; j < columnasImagen; j++){
			suma = 0;
			for (int k = 0; k < tamKernel; k++){
				for (int l = 0; l < tamKernel; l++){
					indiceAmpi = i + k;
					indiceAmpj = j + l;
					suma += imagenAmpleada.at<uchar>(Point(indiceAmpj, indiceAmpi)) * kernel.at<float>(Point(l, k));
					// cout << suma << endl;
				}
			}
			// cout << "Resultado kernel: " << suma << "\n" << endl;
			// imagenFiltroGauss.at<uchar>(Point(i, j)) = uchar(resultadoKernel / sumaKernel);
			conFiltro.at<uchar>(Point(j, i)) = abs(int(suma));
		}
	}

	return conFiltro;
}

Mat expandirImg(Mat imagen, int tamKernel){

	int tamExpancion = ((tamKernel - 1) / 2) * 2;
	cout << "Numero de filas y columnas a expandir: " << tamExpancion << "\n" << endl;

	int filasOrig = imagen.rows; 
	int columnasOrig = imagen.cols; 
	int filasAmp = filasOrig + tamExpancion;
	int columnasAmp = filasOrig + tamExpancion;

	//Imprimimos las dimensiones de las imagenes 
	cout << "Tamaño imagen original: " << filasOrig << "x" << columnasOrig << endl;
	cout << "Tamaño imagen ampleada: " << filasAmp << "x" << columnasAmp << "\n" << endl;

	//Nueva matriz para la ampliada
	Mat imagenAmpleada(filasAmp, columnasAmp, CV_8UC1);

	//Llenamos de 0's la matriz
	for (int i = 0; i < filasAmp; i++){
		for (int j = 0; j < columnasAmp; j++){
			imagenAmpleada.at<uchar>(Point(j, i)) = uchar(0);
		}
	}

	//Pasamos los valores de la img original a la de expansion
	for (int i = 0; i < filasOrig; i++){
		for (int j = 0; j < columnasOrig; j++){
			imagenAmpleada.at<uchar>(Point(j + (tamExpancion / 2), i + (tamExpancion / 2))) = imagen.at<uchar>(Point(j, i));
		}
	}
	// cout << imagenAmpleada << "\n" << endl;

	return imagenAmpleada;
}

Mat baseGx(){
	Mat kernelGx(3, 3, CV_32F);
	kernelGx.at<float>(Point(0, 0)) = -1;
	kernelGx.at<float>(Point(1, 0)) = 0;
	kernelGx.at<float>(Point(2, 0)) = 1;

	kernelGx.at<float>(Point(0, 1)) = -2;
	kernelGx.at<float>(Point(1, 1)) = 0;
	kernelGx.at<float>(Point(2, 1)) = 2;

	kernelGx.at<float>(Point(0, 2)) = -1;
	kernelGx.at<float>(Point(1, 2)) = 0;
	kernelGx.at<float>(Point(2, 2)) = 1;

	return kernelGx;
}

Mat baseGy(){
	Mat kernelGy(3, 3, CV_32F);
	kernelGy.at<float>(Point(0, 0)) = -1;
	kernelGy.at<float>(Point(1, 0)) = -2;
	kernelGy.at<float>(Point(2, 0)) = -1;

	kernelGy.at<float>(Point(0, 1)) = 0;
	kernelGy.at<float>(Point(1, 1)) = 0;
	kernelGy.at<float>(Point(2, 1)) = 0;

	kernelGy.at<float>(Point(0, 2)) = 1;
	kernelGy.at<float>(Point(1, 2)) = 2;
	kernelGy.at<float>(Point(2, 2)) = 1;

	return kernelGy;
}

Mat supresion(Mat sobel, Mat imagenFiltroGauss, Mat matrizGx, Mat matrizGy){

	//Obtenemos la matriz de ºG
	Mat angG(imagenFiltroGauss.rows, imagenFiltroGauss.cols, CV_8UC1);
	Mat bordeDelgado(imagenFiltroGauss.rows, imagenFiltroGauss.cols, CV_8UC1);
	double angulo = 0;
	double vec1 = 255;
	double vec2 = 255;
	double pi = 3.141592;

	for (int i = 0; i < imagenFiltroGauss.rows; i++){
		for (int j = 0; j < imagenFiltroGauss.cols; j++){
			//Formula para sacar ºG
			angulo = atan2(matrizGx.at<uchar>(Point(j, i)), matrizGy.at<uchar>(Point(j, i))) * 180 / pi;
			angG.at<uchar>(Point(j, i)) = angulo;

			if ((0 <= angulo < 22.5) || (157.5 <= angulo <= 180)){
				vec1 = sobel.at<uchar>(Point(j + 1, (i - 1) + 1));
				vec2 = sobel.at<uchar>(Point(j + 1, (i + 1) + 1));
			}
			else if ((67.5 <= angulo < 112.5)) {
				vec1 = sobel.at<uchar>(Point((j - 1) + 1, i + 1));
				vec2 = sobel.at<uchar>(Point((j + 1)+ 1, i + 1));
			}
			else if ((22.5 <= angulo < 67.5)) {
				vec1 = sobel.at<uchar>(Point((j - 1) + 1, (i + 1) + 1));
				vec2 = sobel.at<uchar>(Point((j + 1) + 1, (i - 1) + 1));
			}
			else if ((112.5 <= angulo < 157.5)) {
				vec1 = sobel.at<uchar>(Point((j - 1) + 1, (i - 1) + 1));
				vec2 = sobel.at<uchar>(Point((j + 1) + 1, (i + 1) + 1));
			}

			if (sobel.at<uchar>(Point(j + 1, i + 1)) >= vec1 && sobel.at<uchar>(Point(j + 1, i + 1)) >= vec2) {
				bordeDelgado.at<uchar>(Point(j, i)) = sobel.at<uchar>(Point(j + 1, i + 1));
			}
			else {
				bordeDelgado.at<uchar>(Point(j, i)) = 0;
			}
		}
	}

	return bordeDelgado;

}

Mat umbralizado(Mat sup, float umbralBajo , float umbralAlto) {
	//Matriz donde guardaremos los pixeles fuertes y debiles
	Mat umbrales(sup.rows, sup.cols, CV_8UC1);

	//Busca maximos y minimos en la imagen
	double min, max;
	cv::minMaxLoc(sup, &min, &max);

	//limite superior e inferior
	double limiteSup = max * umbralAlto;
	double limiteInf = limiteSup * umbralBajo;
	Mat matThreshold(sup.rows, sup.cols, CV_8UC1);

	//variables dadas
	int  debil = 25;
	int fuerte = 130;

	for (int i = 1; i < sup.rows - 1; i++) {
		for (int j = 1; j < sup.cols - 1; j++) {


			//Limites
			if (sup.at<uchar>(Point(j, i)) >= limiteSup) {
				umbrales.at<uchar>(Point(j, i)) = fuerte;
			}
			if (sup.at<uchar>(Point(j, i)) <= limiteSup && sup.at<uchar>(Point(j, i)) >= limiteInf) {
				umbrales.at<uchar>(Point(j, i)) = debil;

				//Histeresis
				if (sup.at<uchar>(Point(j + 1, i - 1)) == fuerte || sup.at<uchar>(Point(j + 1, i)) == fuerte
					|| sup.at<uchar>(Point(j + 1, i + 1)) == fuerte || sup.at<uchar>(Point(j, i - 1)) == fuerte
					|| sup.at<uchar>(Point(j, i + 1)) == fuerte || sup.at<uchar>(Point(j - 1, i - 1)) == fuerte
					|| sup.at<uchar>(Point(j - 1, i)) == fuerte || sup.at<uchar>(Point(j - 1, i + 1)) == fuerte) {
					umbrales.at<uchar>(Point(j, i)) = fuerte;
				}
				else {
					umbrales.at<uchar>(Point(j, i)) = 0;
				}

			}
			if (sup.at<uchar>(Point(j, i)) <= limiteInf) {
				umbrales.at<uchar>(Point(j, i)) = 0;
			}


		}
	}



	return umbrales;
}





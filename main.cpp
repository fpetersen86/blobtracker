#include <QtGui/QApplication>
#include "webcamtest.h"
#include <sys/resource.h>

int main(int argc, char** argv)
{

//     const rlim_t kStackSize = 640L * 1024L * 1024L;   // min stack size = 64 Mb
//     struct rlimit rl;
//     int result;
// 
//     getrlimit(RLIMIT_STACK, &rl);
// 	qDebug() << "Current Stack Size = " << rl.rlim_max;
// 	if (rl.rlim_cur < kStackSize)
// 	{
// 		rl.rlim_cur = kStackSize;
// 		result = setrlimit(RLIMIT_STACK, &rl);
// 		if (result != 0)
// 			qDebug() << "setrlimit returned result = " << result;
// 		
// 	} 
// 	getrlimit(RLIMIT_STACK, &rl);
// 	qDebug() << "Current Stack Size = " << rl.rlim_max;
    QApplication app(argc, argv);
 	webcamtest foo;
 	foo.show();
	return app.exec();
}

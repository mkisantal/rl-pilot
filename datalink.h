#ifndef DATALINK_H_
#define DATALINK_H_


#include <mutex>

#include "udp.h"





// TODO: redefine message structs!
// TODO: define pprzlink message
union SlamDunk2PPRZPackage {
    struct  {
        float target_phi_DOT;
        uint8_t status;
    }__attribute__((__packed__));
    unsigned char buf[];
};

struct PPRZ2SlamDunkPackage{
    float heading;
    float psi_dot;
    //unsigned char enables;
} __attribute__((__packed__));




class DataLink{
public:
	void GetMeasurements(float *heading, float *psi_dot);
	void GiveCommand(int action);
	int InitDataLink();

private:

	unsigned char buffer[128];
	std::mutex send_lock;

};

#endif // DATALINK_H_
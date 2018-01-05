#include "datalink.h"
//#include "pprz.h"

#include <iostream>

void DataLink::GetMeasurements(){

	/* Receiving messages (heading, velocity)from the drone. */
/*
	PPRZ2SlamDunkPackage pprz2slamdunk = {0};
	bool msg_available = true;
	//receive all available packages, throw away old ones:
	while (msg_available) {
		msg_available = false;
		pprz_.tryReceive(&pprz2slamdunk,&msg_available);
	}

    float heading = pprz2slamdunk.heading;
    float psi_dot = pprz2slamdunk.psi_dot;
*/
}

void DataLink::GiveCommand(){

	/* Giving steering commands to the drone. */
/*	
	SlamDunk2PPRZPackage slamdunk2pprz = {0};

	slamdunk2pprz.status = 0;
    pprz_.writePackage(&slamdunk2pprz);
*/
}

int DataLink::InitDataLink(){

	/* Initializing paparazzi - slamdunk communication. */
/*

	if (pprz_.init(13374)) {
		std::cout << "PPRZ link problem" << std::endl;
		return 1;
	}

	return 0;
*/
}
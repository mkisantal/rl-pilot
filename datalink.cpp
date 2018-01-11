#include "datalink.h"

#include <iostream>

#include "pprzlink/pprz_transport.h"
#include "pprzlink/intermcu_msg.h"

#include "udp.h"

struct pprz_transport trans;

void DataLink::GetMeasurements(float *heading, float *psi_dot){

	/* Receiving messages (heading, velocity)from the drone. */
	
	std::cout << "GetMeasurements" << std::endl;

	PPRZ2SlamDunkPackage p2s_package = {0};
	bool msg_available = true;
	//receive all available packages, throw away old ones:
	while (msg_available) {
		msg_available = false;

		pprz_check_and_parse(&(udp0.device),&trans,buffer,&msg_available);

  		if (msg_available) {
    		unsigned char * tmp = (unsigned char *) p2s_package;
    		for(int i=0; i<sizeof(struct PPRZ2SlamDunkPackage);i++) {
      			tmp[i] = buffer[i+3];	//TODO!!!! offset
    		}
  		}
	}

    *heading = p2s_package.heading;
    *psi_dot = p2s_package.psi_dot;


}

void DataLink::GiveCommand(int action){

	/* Giving steering commands to the drone. */

	std::cout << "GiveCommand: " << action << std::endl;


	SlamDunk2PPRZPackage s2p_package = {0};

	s2p_package.status = 0;

    send_lock.lock();	// TODO: a message has to be defined in pprzlink!!!
  		pprz_msg_send_PAYLOAD(&(trans.trans_tx), &(udp0.device), 3, sizeof(SlamDunk2PPRZPackage),
    s2p_package.buf);
  	send_lock.unlock();

}

int DataLink::InitDataLink(){

	/* Initializing paparazzi - slamdunk communication. */

	std::cout << "InitDataLink" << std::endl;

	pprz_transport_init(&trans);
	UDP0Init();

	return 0;
}
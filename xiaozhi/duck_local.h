#ifndef DUCK_LOCAL_H
#define DUCK_LOCAL_H

namespace duck_local {

// Kitchen timer tools for the cloud model: self.timer.start / status / cancel.
void AddTools();

// Local voice command "ting xia": tells the duck to stop over the LAN, no cloud.
void StopDuck();

}  // namespace duck_local

#endif

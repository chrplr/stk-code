//
//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2026 SuperTuxKart-Team
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include "gym/gym_controller.hpp"

#include "graphics/camera/camera.hpp"
#include "guiengine/engine.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/rescue_animation.hpp"
#include "modes/world.hpp"

//-----------------------------------------------------------------------------
GymController::GymController(AbstractKart *kart) : Controller(kart)
{
    setControllerName("GymController");

    // LocalPlayerController makes its own camera in the constructor and
    // ProfileWorld makes one for the kart it wants to watch. This controller is
    // not a local player, so nothing else would create one, and the windowed
    // mode would render a race from nowhere.
    if (!GUIEngine::isNoGraphics())
        Camera::createCamera(kart, 0);
}   // GymController

//-----------------------------------------------------------------------------
void GymController::reset()
{
    m_action.reset();
    m_controls->reset();
}   // reset

//-----------------------------------------------------------------------------
/** Applies the current action. This runs on every physics tick, so one agent
 *  step that covers several ticks holds its action for all of them.
 *  \param ticks Number of physics steps - should be 1.
 */
void GymController::update(int ticks)
{
    if (World::getWorld()->isStartPhase())
    {
        // Touching the controls before GO earns a start penalty. An agent
        // cannot see that coming, so simply hold still: GymServer steps through
        // the start phase itself before returning the first observation.
        m_controls->reset();
        return;
    }
    m_controls->setSteer(m_action.m_steer);
    m_controls->setAccel(m_action.m_accel);
    m_controls->setBrake(m_action.m_brake);
    m_controls->setNitro(m_action.m_nitro);
    m_controls->setFire(m_action.m_fire);
    m_controls->setSkidControl((KartControl::SkidControl)m_action.m_skid);

    // Rescue is edge triggered: the animation is started once and the request
    // cleared, exactly as PlayerController::update does it, otherwise a held
    // rescue would restart the animation on every tick.
    if (m_action.m_rescue)
    {
        if (!m_kart->getKartAnimation())
        {
            RescueAnimation::create(m_kart);
            m_action.m_rescue = false;
        }
        m_controls->setRescue(false);
    }
}   // update

//-----------------------------------------------------------------------------
/** The gym server drives a local, single-process race; there is no rewinding to
 *  do, so these are the same no-ops GhostController uses. */
bool GymController::saveState(BareNetworkString *buffer) const
{
    return false;
}   // saveState

//-----------------------------------------------------------------------------
void GymController::rewindTo(BareNetworkString *buffer)
{
}   // rewindTo

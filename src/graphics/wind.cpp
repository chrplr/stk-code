//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2014-2015 SuperTuxKart-Team
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

#include "graphics/irr_driver.hpp"
#include "graphics/wind.hpp"
#include "utils/helpers.hpp"

#include <IrrlichtDevice.h>

Wind::Wind()
{
    m_seed = 0.0f;
    m_time = 0.0f;
}

void Wind::reseed(int seed)
{
    m_seed = (float)((seed % 1000) - 500);
    m_time = 0.0f;
}

vector3df Wind::getWind() const
{
    return m_wind;
}

void Wind::update(float dt)
{
    vector3df dir(0, 0, 1);
    m_time += dt;
    const float now = m_time * 1000.0f;

    const float rotation = asinf(noise2d(m_seed, now / 100000.0f)) * RAD_TO_DEGREE;
    dir.rotateXZBy(rotation);

    m_wind = dir;
}

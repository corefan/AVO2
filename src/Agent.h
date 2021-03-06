/*
 * Agent.h
 * AVO2 Library
 *
 * Copyright 2010 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Jamie Snape, Stephen J. Guy, and Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/AVO/>
 */

#ifndef AVO_AGENT_H_
#define AVO_AGENT_H_

/**
 * \file   Agent.h
 * \brief  Declares the Agent class.
 */

#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include "Line.h"
#include "Vector2.h"

namespace AVO {
class KdTree;

/**
 * \class  Agent
 * \brief  An agent in the simulation.
 */
class Agent {
 private:
  /**
   * \brief  Constructor.
   */
  Agent()
      : id_(0),
        maxNeighbors_(0),
        accelInterval_(0.0f),
        maxAccel_(0.0f),
        maxSpeed_(0.0f),
        neighborDist_(0.0f),
        radius_(0.0f),
        timeHorizon_(0.0f) {}

  /**
   * \brief      Computes the neighbors of this agent.
   * \param[in]  kdTree  A k-D tree containing the neighbors.
   */
  void computeNeighbors(const KdTree *kdTree);

  /**
   * \brief      Computes the new velocity of this agent.
   * \param[in]  timeStep  The time step of the simulation.
   */
  void computeNewVelocity(float timeStep);

  /**
   * \brief          Inserts a neighbor into the set of neighbors of this agent.
   * \param[in]      agent    The agent to be inserted.
   * \param[in,out]  rangeSq  The squared range around this agent.
   */
  void insertAgentNeighbor(const Agent *agent, float &rangeSq);

  /**
   * \brief      Updates the position and velocity of this agent.
   * \param[in]  timeStep  The time step of the simulation.
   */
  void update(float timeStep) {
    velocity_ = newVelocity_;
    position_ += velocity_ * timeStep;
  }

  // Not implemented.
  Agent(const Agent &other);

  // Not implemented.
  Agent &operator=(const Agent &other);

  Vector2 newVelocity_;
  Vector2 position_;
  Vector2 prefVelocity_;
  Vector2 velocity_;
  std::size_t id_;
  std::size_t maxNeighbors_;
  float accelInterval_;
  float maxAccel_;
  float maxSpeed_;
  float neighborDist_;
  float radius_;
  float timeHorizon_;
  std::deque<Vector2> boundary_;
  std::vector<std::pair<float, const Agent *> > agentNeighbors_;
  std::vector<Line> orcaLines_;

  friend class KdTree;
  friend class Simulator;
};

/**
 *  \brief          Solves a one-dimensional linear program on a specified line
 *                  subject to linear constraints defined by lines and a
 *                  circular constraint.
 *  \param[in]      lines         Lines defining the linear constraints.
 *  \param[in]      lineNo        The specified line constraint.
 *  \param[in]      radius        The radius of the circular constraint.
 *  \param[in]      optVelocity   The optimization velocity.
 *  \param[in]      directionOpt  True if the direction should be optimized.
 *  \param[in,out]  result        A reference to the result of the linear
 *                                program.
 *  \return         True if successful.
 */
bool linearProgram1(const std::vector<Line> &lines, std::size_t lineNo,
                    float radius, const Vector2 &optVelocity, bool directionOpt,
                    Vector2 &result);

/**
 *  \brief          Solves a two-dimensional linear program subject to linear
 *                  constraints defined by lines and a circular constraint.
 *  \param[in]      lines         Lines defining the linear constraints.
 *  \param[in]      radius        The radius of the circular constraint.
 *  \param[in]      optVelocity   The optimization velocity.
 *  \param[in]      directionOpt  True if the direction should be optimized.
 *  \param[in,out]  result        A reference to the result of the linear
 *                                program.
 *  \return         The number of the line it fails on, and the number of lines
 *                  if successful.
 */
std::size_t linearProgram2(const std::vector<Line> &lines, float radius,
                           const Vector2 &optVelocity, bool directionOpt,
                           Vector2 &result);

/**
 *  \brief          Solves a two-dimensional linear program subject to linear
 *                  constraints defined by lines and a circular constraint.
 *  \param[in]      numObstLines  Number of obstacle constraints.
 *  \param[in]      lines         Lines defining the linear constraints.
 *  \param[in]      radius        The radius of the circular constraint.
 *  \param[in,out]  result        A reference to the result of the linear
 *                                program.
 */
void linearProgram3(const std::vector<Line> &lines, std::size_t numObstLines,
                    float radius, Vector2 &result);
}  // namespace AVO

#endif  // AVO_AGENT_H_

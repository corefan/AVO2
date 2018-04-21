/*
 * Simulator.cc
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

#include "Simulator.h"

#include <limits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Agent.h"
#include "KdTree.h"

namespace AVO {
Simulator::Simulator()
    : defaultAgent_(NULL), kdTree_(NULL), globalTime_(0.0f), timeStep_(0.0f) {
  kdTree_ = new KdTree(this);
}

Simulator::~Simulator() {
  delete defaultAgent_;

  for (std::size_t agentNo = 0; agentNo < agents_.size(); ++agentNo) {
    delete agents_[agentNo];
  }

  delete kdTree_;
}

std::size_t Simulator::addAgent(const Vector2 &position) {
  if (defaultAgent_ == NULL) {
    return std::numeric_limits<std::size_t>::max();
  }

  Agent *agent = new Agent();

  agent->accelInterval_ = defaultAgent_->accelInterval_;
  agent->maxAccel_ = defaultAgent_->maxAccel_;
  agent->maxNeighbors_ = defaultAgent_->maxNeighbors_;
  agent->maxSpeed_ = defaultAgent_->maxSpeed_;
  agent->neighborDist_ = defaultAgent_->neighborDist_;
  agent->position_ = position;
  agent->radius_ = defaultAgent_->radius_;
  agent->timeHorizon_ = defaultAgent_->timeHorizon_;
  agent->velocity_ = defaultAgent_->velocity_;
  agent->id_ = agents_.size();

  agents_.push_back(agent);

  return agents_.size() - 1;
}

std::size_t Simulator::addAgent(const Vector2 &position, float neighborDist,
                                std::size_t maxNeighbors, float timeHorizon,
                                float radius, float maxSpeed, float maxAccel,
                                float accelInterval) {
  Agent *agent = new Agent();

  agent->accelInterval_ = accelInterval;
  agent->maxAccel_ = maxAccel;
  agent->maxNeighbors_ = maxNeighbors;
  agent->maxSpeed_ = maxSpeed;
  agent->neighborDist_ = neighborDist;
  agent->position_ = position;
  agent->radius_ = radius;
  agent->timeHorizon_ = timeHorizon;
  agent->id_ = agents_.size();

  agents_.push_back(agent);

  return agents_.size() - 1;
}

std::size_t Simulator::addAgent(const Vector2 &position, float neighborDist,
                                std::size_t maxNeighbors, float timeHorizon,
                                float radius, float maxSpeed, float maxAccel,
                                float accelInterval, const Vector2 &velocity) {
  Agent *agent = new Agent();

  agent->accelInterval_ = accelInterval;
  agent->maxAccel_ = maxAccel;
  agent->maxNeighbors_ = maxNeighbors;
  agent->maxSpeed_ = maxSpeed;
  agent->neighborDist_ = neighborDist;
  agent->position_ = position;
  agent->radius_ = radius;
  agent->timeHorizon_ = timeHorizon;
  agent->velocity_ = velocity;
  agent->id_ = agents_.size();

  agents_.push_back(agent);

  return agents_.size() - 1;
}

void Simulator::doStep() {
  kdTree_->buildAgentTree();

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int agentNo = 0; agentNo < static_cast<int>(agents_.size()); ++agentNo) {
    agents_[agentNo]->computeNeighbors(kdTree_);
    agents_[agentNo]->computeNewVelocity(timeStep_);
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int agentNo = 0; agentNo < static_cast<int>(agents_.size()); ++agentNo) {
    agents_[agentNo]->update(timeStep_);
  }

  globalTime_ += timeStep_;
}

float Simulator::getAgentAccelInterval(std::size_t agentNo) const {
  return agents_[agentNo]->accelInterval_;
}

std::size_t Simulator::getAgentNeighbor(std::size_t agentNo,
                                        std::size_t neighborNo) const {
  return agents_[agentNo]->agentNeighbors_[neighborNo].second->id_;
}

float Simulator::getAgentMaxAccel(std::size_t agentNo) const {
  return agents_[agentNo]->maxAccel_;
}

std::size_t Simulator::getAgentMaxNeighbors(std::size_t agentNo) const {
  return agents_[agentNo]->maxNeighbors_;
}

float Simulator::getAgentMaxSpeed(std::size_t agentNo) const {
  return agents_[agentNo]->maxSpeed_;
}

float Simulator::getAgentNeighborDist(std::size_t agentNo) const {
  return agents_[agentNo]->neighborDist_;
}

std::size_t Simulator::getAgentNumNeighbors(std::size_t agentNo) const {
  return agents_[agentNo]->agentNeighbors_.size();
}

std::size_t Simulator::getAgentNumOrcaLines(std::size_t agentNo) const {
  return agents_[agentNo]->orcaLines_.size();
}

const Line &Simulator::getAgentOrcaLine(std::size_t agentNo,
                                        std::size_t lineNo) const {
  return agents_[agentNo]->orcaLines_[lineNo];
}

const Vector2 &Simulator::getAgentPosition(std::size_t agentNo) const {
  return agents_[agentNo]->position_;
}

const Vector2 &Simulator::getAgentPrefVelocity(std::size_t agentNo) const {
  return agents_[agentNo]->prefVelocity_;
}

float Simulator::getAgentRadius(std::size_t agentNo) const {
  return agents_[agentNo]->radius_;
}

float Simulator::getAgentTimeHorizon(std::size_t agentNo) const {
  return agents_[agentNo]->timeHorizon_;
}

const Vector2 &Simulator::getAgentVelocity(std::size_t agentNo) const {
  return agents_[agentNo]->velocity_;
}

void Simulator::setAgentAccelInterval(std::size_t agentNo,
                                      float accelInterval) {
  agents_[agentNo]->accelInterval_ = accelInterval;
}

void Simulator::setAgentDefaults(float neighborDist, std::size_t maxNeighbors,
                                 float timeHorizon, float radius,
                                 float maxSpeed, float maxAccel,
                                 float accelInterval, const Vector2 &velocity) {
  if (defaultAgent_ == NULL) {
    defaultAgent_ = new Agent();
  }

  defaultAgent_->accelInterval_ = accelInterval;
  defaultAgent_->maxAccel_ = maxAccel;
  defaultAgent_->maxNeighbors_ = maxNeighbors;
  defaultAgent_->maxSpeed_ = maxSpeed;
  defaultAgent_->neighborDist_ = neighborDist;
  defaultAgent_->radius_ = radius;
  defaultAgent_->timeHorizon_ = timeHorizon;
  defaultAgent_->velocity_ = velocity;
}

void Simulator::setAgentMaxAccel(std::size_t agentNo, float maxAccel) {
  agents_[agentNo]->maxAccel_ = maxAccel;
}

void Simulator::setAgentMaxNeighbors(std::size_t agentNo,
                                     std::size_t maxNeighbors) {
  agents_[agentNo]->maxNeighbors_ = maxNeighbors;
}

void Simulator::setAgentMaxSpeed(std::size_t agentNo, float maxSpeed) {
  agents_[agentNo]->maxSpeed_ = maxSpeed;
}

void Simulator::setAgentNeighborDist(std::size_t agentNo, float neighborDist) {
  agents_[agentNo]->neighborDist_ = neighborDist;
}

void Simulator::setAgentPosition(std::size_t agentNo, const Vector2 &position) {
  agents_[agentNo]->position_ = position;
}

void Simulator::setAgentPrefVelocity(std::size_t agentNo,
                                     const Vector2 &prefVelocity) {
  agents_[agentNo]->prefVelocity_ = prefVelocity;
}

void Simulator::setAgentRadius(std::size_t agentNo, float radius) {
  agents_[agentNo]->radius_ = radius;
}

void Simulator::setAgentTimeHorizon(std::size_t agentNo, float timeHorizon) {
  agents_[agentNo]->timeHorizon_ = timeHorizon;
}

void Simulator::setAgentVelocity(std::size_t agentNo, const Vector2 &velocity) {
  agents_[agentNo]->velocity_ = velocity;
}
}  // namespace AVO

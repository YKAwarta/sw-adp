# Interview Prep: Datasets for SW-ADP Project

To prepare for an interview, you want to be able to explain these datasets simply. The core idea of your project is **"We are taking internal satellite health data and matching it with external weather data to see if weather caused the health issues."**

Here is a breakdown of the three datasets in an easy-to-understand way:

---

## 1. OPSSAT-AD (ESA)

**What is this in simple terms?**
Imagine you are a mechanic looking at a car's internal dashboard (engine temperature, battery voltage, oil pressure). This dataset is exactly that, but for a satellite. It is the **"internal health record"** of the European Space Agency's (ESA) _OPS-SAT_ CubeSat.

**Major Fields / Data Samples:**
It contains 9 "telemetry channels" (sensors reading internal data). Examples include:

- **Voltages & Currents:** To see if the power system is stable.
- **Temperatures:** To check if parts are overheating.
- **Magnetic Field Readings:** How the satellite is oriented.
- **Ground Truth Labels:** Crucially, engineers have manually marked blocks of this data as "Normal" or "Anomaly" (meaning something broke or acted weird).

**How it helps the project:**
This is your **baseline dataset**. You are using this to see when the satellite actually experienced problems. Your AI model will try to predict the "Anomaly" labels based on the sensor readings.

---

## 2. NOAA SWPC (Space Weather Prediction Center)

**What is this in simple terms?**
If OPSSAT is the "car dashboard," NOAA SWPC is the **"external weather report"** (e.g., is it raining, snowing, or a hurricane?). Instead of rain, we are talking about _Space Weather_—solar flares, radiation, and magnetic storms coming from the Sun.

**Major Fields / Data Samples:**

- **Kp Index (Planetary K-index):** A scale from 0 to 9 measuring how disturbed Earth's magnetic field is. High number = big storm.
- **Dst Index:** Measures the strength of a space storm based on Earth's magnetic ring current.
- **F10.7 (Solar Radio Flux):** A number representing how generally "active" the Sun is on any given day.
- **Solar Wind Speed/Density:** How fast and thick the stream of particles from the sun is hitting Earth.

**How it helps the project:**
This is your **"secret sauce."** By taking the timestamp of an anomaly from the OPSSAT dataset and looking up what the Sun was doing at that _exact same time_ in the NOAA dataset, your AI can learn: _"Oh, whenever the Kp Index spikes or the solar wind is fast, the satellite's voltage drops."_ It provides the **cause** for the anomaly.

---

## 3. ESA Anomaly Dataset (Multi-Mission)

**What is this in simple terms?**
Imagine you built a diagnostic tool that perfectly predicts problems on a _Honda Civic_ (the OPS-SAT). Will your tool also work on a _Ford truck_ or a _Toyota SUV_? The ESA Anomaly Dataset gives you data from **other types of satellites (three different missions over 9 years)** to test your tool.

**Major Fields / Data Samples:**

- It contains 31 GB of telemetry (health data, like OPSSAT) but from different, much larger flagship missions rather than just one small CubeSat.
- It also contains expert-annotated anomaly logs.

**How it helps the project:**
This is used for **"Generalization Testing."** In an interview, if they ask: _"How do you know your AI doesn't just memorize the OPS-SAT satellite specifically?"_ You answer: _"After I prove my model works on OPS-SAT, I will test it on the ESA Anomaly Dataset to prove that my Space Weather model works on entirely different satellites too, proving it is a universal solution."_

---

### Interview Summary (The "Elevator Pitch")

_"We use **OPSSAT-AD** to see when a specific satellite broke down. We use **NOAA SWPC** to see what the weather in space was like at that exact moment. By combining them, we teach an AI to realize that space weather causes satellite failures, allowing us to predict problems faster and with fewer false alarms. Finally, we use the **ESA Anomaly Dataset** to prove our AI works on any satellite, not just one."_

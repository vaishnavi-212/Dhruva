# Field test — KLE Tech to Unkal Lake

Everything below runs on the phone with no internet, except downloading the map tiles once, at a desk.

## 1. Build and install the APK

On the laptop, in the repo folder:

```
git checkout main && git pull
./gradlew :app:assembleDebug
```

The file appears at `app/build/outputs/apk/debug/app-debug.apk`.

With the phone plugged in and USB debugging on:

```
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Or copy the APK to the phone and open it (allow "install unknown apps" once).

Grant **Location: while using the app** and **precise location** when asked.

## 2. The day before, at a desk on wifi

1. Open the app, tap **Open Navigate Screen**.
2. Pan and zoom so campus, Pune-Bangalore Road and Unkal Lake are all on screen.
3. Tap **Save map offline** and wait for "Map saved for offline use". Without this the map is grey
   with no signal. Nothing else needs the internet: places, routing, the AI model and the voice are
   all on the phone.

## 3. Before every ride

- Phone **mounted on the handlebar**, screen on, portrait. Pocket rides score much worse (RESOURCES 8ap).
- Sit still for **3 seconds** before you cut GPS. The status line must not say "gyro bias NOT measured".
- Wait for the status line to show **Mode: GNSS**.
- Start the phone's **screen recorder**. The recording is the evidence for the PPT.
- Tap **Where to?**, type the destination, pick it from the list. The button must show
  "→ Unakal Kere · 1.5 km · 3 min" and the line under the map "bound to Unakal Kere".

## 4. The three rides

| # | Ride | What to do | What should happen |
|---|---|---|---|
| 1 | Sanity, GPS on | Ride to the lake normally | "Turn right in ... metres" at the gate, "You have arrived at Unakal Kere" |
| 2 | The demo | After 100-150 m on campus, flip **Simulate GPS loss** ON and leave it on | "Satellite signal lost", the dot keeps moving on the route, the voice says "about" instead of exact distances, arrival announced |
| 3 | Wrong turn | Cut GPS before the gate, then turn **left** instead of right | "Wrong turn" within ~20-40 m, then "Re-routing" a few seconds later, and guidance continues on the new route |

After each ride: tap **Finish Run**, read the card, then turn the simulated loss OFF so GPS comes back.

## 5. What to bring back

- The **screen recordings**.
- The files in `Android/data/com.dhruva.nav/files/`: `trips/trip_*.json`, `perf_*.csv`, and the ride folder
  if recording was on. Share them with "Share Last Run" or copy over USB.
- A photo of the **Finish Run** card for each ride.

## 6. What the numbers should look like

From the campus benchmark and the recorded rides:

- **Drift with GPS off**: 3-7% of the distance ridden. Under 10% is the ISRO limit. Ride 2 is the one that matters.
- **Arrival**: the card's final error is how far the dot was from the true end, tens of metres, not hundreds.
- **Wrong turn**: caught a median of 19 m after leaving the route; up to ~80 m is still fine.
- **AI speed**: about 15 ms per run on a real phone, memory around 150 MB, no measurable battery drop in 10 minutes.

## 7. If something goes wrong

Write down **what the screen said and when**. The status line under the map is the fastest clue:

- "gyro bias NOT measured" — you cut GPS too soon after starting.
- "No route for this area" — the route was not bound; re-plan the destination with GPS on.
- "Off route — no side road found here" — a wrong turn was noticed but there was no other road to move to.
- The dot does not move during a blackout — the app had no moving speed before the cut. Ride at least
  50 m with GPS on first.

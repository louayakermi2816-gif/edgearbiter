# Node-RED with the Dashboard 2.0 palette and the EdgeArbiter flow (control loop + board at /dashboard).
FROM nodered/node-red:5.0.9
RUN npm install --no-audit --no-fund --omit=dev @flowfuse/node-red-dashboard@1.34.0
COPY flows.json /data/flows.json

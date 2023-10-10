from locust import HttpUser, task, between
import requests


class YOLOUser(HttpUser):
    wait_time = between(1, 5)  # Time between requests

    @task
    def detect_objects(self):
        # Replace with the actual URL of your YOLO model endpoint
        url = "http://localhost:8000/yolo-speed-check"

        # Load an image from your test dataset
        # Replace with your image loading logic
        image_path = "for_testing/17299.png"
        data = {'images': ('image1.png', open(image_path, 'rb'))}

        # Send a POST request with the image to your YOLO model
        headers = {"Content-Type": "image/jpeg"}
        response = self.client.post(url, files=data)

        # Check the response status code and content
        if response.status_code == 200:
            # Optionally, parse and validate the YOLO model's output
            # For object detection, the response may contain bounding boxes and labels
            detected_objects = response.json()
            if len(detected_objects) > 0:
                self.environment.events.request.fire(
                    request_type="object_detection",
                    name="detect_objects",
                    response_time=response.elapsed.total_seconds(),
                    response_length=len(response.content),
                )
            else:
                self.environment.events.request.fire(
                    request_type="object_detection",
                    name="detect_objects",
                    response_time=response.elapsed.total_seconds(),
                    response_length=len(response.content),
                    exception="No objects detected",
                )
        else:
            self.environment.events.request.fire(
                request_type="object_detection",
                name="detect_objects",
                response_time=response.elapsed.total_seconds(),
                response_length=len(response.content),
                exception="Failed to detect objects",
            )

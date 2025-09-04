from bounding_box import BBox
import cv2, logging

class MouseSelector:
    """
    Selecciona/deselecciona objetos mediante clic.
    Guarda la BBox elegida para el modo focus.
    """
    def __init__(self):
        self.selected_object: BBox | None = None

    def mouse_callback(self, event, x, y, flags, boxes):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pt = (x, y)
        # ¿Había algo ya seleccionado?
        if self.selected_object and self.selected_object.contains(*pt):
            logging.info("Deseleccionando objeto.")
            self.selected_object = None
            return

        # Buscar la primera caja que contenga el clic
        for box in boxes:
            if box.contains(*pt):
                self.selected_object = box
                logging.info(f"Objeto seleccionado: {self.selected_object}")
                break
        else:
            logging.info("Clic fuera de todas las cajas.")